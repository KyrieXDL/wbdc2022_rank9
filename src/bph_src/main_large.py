# encoding=utf-8
import copy
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from config import parse_args
from dataset.data_helper import create_dataloaders, MultiModalDataset
from models.modelLarge import MultiModal
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate, init_distributed_mode, \
    copy_params, momentum_update
from callback.adversarial import FGM, PGD
import time
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast


def cal_loss(prediction, label):
    label = label.squeeze(dim=1)
    loss = F.cross_entropy(prediction, label)
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]
    return loss, accuracy, pred_label_id, label


def validate(model, val_dataloader, device, args):
    model.eval()
    predictions = []
    labels = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='valing...', total=len(val_dataloader)):
            frame_input, frame_mask = batch['frame_input'].to(device), batch['frame_mask'].to(device)
            title_input, title_mask = batch['title_input'].to(device), batch['title_mask'].to(device)
            token_type_ids, label = batch['token_type_ids'].to(device), batch['label'].to(device)
            prediction = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
            # label转换到cuda
            loss, accuracy, pred_label_id, _ = cal_loss(prediction, label)
            loss = loss.mean()
            predictions.append(pred_label_id)
            labels.append(label.squeeze(1))
            # predictions.extend(pred_label_id)
            # labels.extend(label.squeeze(1))
            # predictions.extend(pred_label_id.cpu().numpy())
            # labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())

    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    if args.distributed:
        predictions_tensor_list = [torch.zeros_like(predictions) for _ in range(dist.get_world_size())]
        labels_tensor_list = [torch.zeros_like(labels) for _ in range(dist.get_world_size())]
        dist.all_gather(predictions_tensor_list, predictions)
        dist.all_gather(labels_tensor_list, labels)
        predictions = torch.cat(predictions_tensor_list, dim=0)
        labels = torch.cat(labels_tensor_list, dim=0)

    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
    device = torch.device(args.device)
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, args.train_zip_feats)
    train_index, val_index = [i for i in range(90000)], [i for i in range(90000, 100000)]
    size = len(dataset)
    print(f"total num: {size}")
    # train_index, val_index = [i for i in range(90000)], [i for i in range(90000, 100000)]
    train_index, val_index = [i for i in range(size - 10000)], [i for i in range(size - 10000, size)]
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    print(len(train_dataset), len(val_dataset))
    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True,
                                  drop_last=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.val_batch_size, pin_memory=True,
                                drop_last=False, num_workers=args.num_workers, prefetch_factor=args.prefetch)

    # 2. build model and optimizers
    model = MultiModal(args)
    model.to(device)
    # for n, p in model.named_parameters():
    #     if p.requires_grad and p.grad is None:
    #         p.requires_grad=False
    
    model_without_ddp = model
    if args.use_ema:
        momentum_model = MultiModal(args).to(device)
        copy_params([model, momentum_model])
        momentum_model.eval()
    else:
        momentum_model = None
    #-------------同步bn层---------------#
    # ngpus_per_node  = torch.cuda.device_count()
    # if ngpus_per_node > 1 and args.distributed:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     print('sync bn success!')
    # elif sync_bn:
    #     print("Sync_bn is not support in one gpu or not distributed.")
    #------------------------------------#
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        
    if args.fp16:
        #------------------------------------------------------------------#
        #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
        #   因此torch1.2这里显示"could not be resolve"
        #------------------------------------------------------------------#
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
        # cudnn.benchmark = True
    else:
        logging.info('we do not use fp16')
        scaler = None
        # cudnn.benchmark = True

    swa_raw_model = copy.deepcopy(model)
    t_total = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, t_total)
    if args.freeze:
        unfreeze_layers = ['layer.6','layer.7','layer.8','layer.9','layer.10','layer.11','layer.12','layer.13','layer.14','layer.15','layer.16','layer.17','layer.18','layer.19','layer.20','layer.21','layer.22','layer.23','bert.pooler','out.']
        for name ,param in model.named_parameters():
            param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break        
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    logging.info(f"start time >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    for epoch in range(args.max_epochs):
        if epoch == args.early_stop:
            break
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        model.train()
        if args.change_lr:
            if epoch >= args.change_epoch:
                for p in optimizer.param_groups:
                    before_lr = p['lr']
                    p['lr'] *= 0.5
                    print(f"before lr={before_lr}, after lr={p['lr']}")
        for batch in tqdm(train_dataloader, desc=f"{epoch}/{args.max_epochs} training...", total=len(train_dataloader)):
            model.train()
            frame_input, frame_mask = batch['frame_input'].to(device), batch['frame_mask'].to(device)
            title_input, title_mask = batch['title_input'].to(device), batch['title_mask'].to(device)
            token_type_ids, label = batch['token_type_ids'].to(device), batch['label'].to(device)
            
            if args.fp16:
                with autocast():
                    prediction, itc_loss = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
                    loss, accuracy, _, _ = cal_loss(prediction, label)
                    if args.contras:
                        loss = (loss + itc_loss).mean()
                    accuracy = accuracy.mean()
            else:
                prediction, itc_loss = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
                loss, accuracy, _, _ = cal_loss(prediction, label)
                if args.contras:
                    loss = (loss + itc_loss).mean()
                accuracy = accuracy.mean()

            if args.use_ema:
                with torch.no_grad():
                    momentum_update([model, momentum_model], args.momentum)
            
            if args.fp16:
                scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
            else:
                loss.backward() 
                # optimizer.step()

            if args.use_fgm:
                # if args.use_fgm:
                # word_embeddings
                fgm = FGM(model, emb_name="word_embeddings", epsilon=args.epsilon)
                fgm.attack()
                if args.fp16:
                    with autocast():
                        prediction, itc_loss = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
                        loss_adv, accuracy, _, _ = cal_loss(prediction, label)
                        if args.contras:
                            loss_adv = (loss_adv + itc_loss).mean()
                    scaler.scale(loss_adv).backward()
                else:
                    prediction, itc_loss = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
                    loss_adv, accuracy, _, _ = cal_loss(prediction, label)
                    if args.contras:
                        loss_adv = (loss_adv + itc_loss).mean()
                    loss_adv.backward()
                fgm.restore()

            if args.use_pgd:
                pgd = PGD(model, emb_name="word_embeddings", epsilon=args.epsilon, alpha=args.alpha)
                pgd.backup_grad()
                for _t in range(args.adv_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != args.adv_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    if args.fp16:
                        with autocast():
                            prediction, itc_loss = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
                            loss_adv, accuracy, _, _ = cal_loss(prediction, label)
                            if args.contras:
                                loss_adv = (loss_adv + itc_loss).mean()
                        scaler.scale(loss_adv).backward()
                    else:
                        prediction, itc_loss = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
                        loss_adv, accuracy, _, _ = cal_loss(prediction, label)
                        if args.contras:
                            loss_adv = (loss_adv + itc_loss).mean()
                        loss_adv.backward()
                pgd.restore()
                    
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # if args.use_ema:
            #     with torch.no_grad():
            #         momentum_update([model, momentum_model], args.momentum)
                    
            optimizer.zero_grad()
            scheduler.step()
            
            step += 1
            if step % args.print_steps == 0 and args.rank == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                print(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}, "
                      f"lr {scheduler.get_last_lr()[0]}")

        # 4. validation
        start_time = time.time()
        loss, results = validate(model, val_dataloader, device, args)
        end_time = time.time()
        total_time = (end_time - start_time) / 10000
        
        # 之后注释掉
        # if args.use_ema and args.rank == 0:
        #     mean_f1 = results['mean_f1']
        #     torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
        #                    f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}_{epoch}_noema.bin')
        
        if args.rank == 0:
            results = {k: round(v, 4) for k, v in results.items()}
            logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}, time: {total_time}")
            print(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}, time: {total_time}")

        if args.use_ema:
            # 4. validation
            loss, results = validate(momentum_model, val_dataloader, device, args)
            if args.rank == 0:
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"EMA Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                print(f"EMA Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint
        if args.rank == 0:
            mean_f1 = results['mean_f1']
            if args.use_ema:
                torch.save({'epoch': epoch, 'model_state_dict': momentum_model.state_dict(), 'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}_{epoch}.bin')
            else:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}_{epoch}.bin')

            if mean_f1 > best_score:
                best_score = mean_f1
            # torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
            #            f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')

#     swa_raw_model = swa(swa_raw_model, args.savedmodel_path, swa_start=args.swa_start)

#     loss, results = validate(swa_raw_model, val_dataloader, device, args)
    
#     if args.rank == 0:
#         results = {k: round(v, 4) for k, v in results.items()}
#         logging.info(f"average swa: loss {loss:.3f}, {results}")
#         print(f"average swa: loss {loss:.3f}, {results}")
#         # 5. save checkpoint
#         mean_f1 = results['mean_f1']
#         torch.save(
#             {'epoch': args.max_epochs + 1, 'model_state_dict': swa_raw_model.state_dict(), 'mean_f1': mean_f1},
#             f'{args.savedmodel_path}/model_epoch_swa_mean_f1_{mean_f1}_swa.bin')

#         logging.info(f"end time >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")


def main():
    args = parse_args()
    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    if args.distributed:
        init_distributed_mode(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
