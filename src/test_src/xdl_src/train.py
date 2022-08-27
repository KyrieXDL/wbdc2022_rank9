import os
import torch
from torch.utils.data import DataLoader
from mydataset.wechat_datatset import Wechat_Dataset
from models.multimodal_classifiser import MultiModal_Classifier
# from models.multimodal_classifiser_v2 import MultiModal_Classifier_V2
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import utils
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
import time
from models.fgm import FGM
from models.pgd import PGD
from loss import ASLSingleLabel, compute_kl_loss, LabelSmoothingCrossEntropy, AsymmetricLossOptimized
from mydataset.category_id_map import category_id_to_lv2id, category_id_to_lv1id, lv2id_to_category_id, lv2id_to_lv1id
import copy

K = 3
cnt = [9, 5, 9, 6, 6, 3, 3, 6, 6, 8, 4, 6, 6, 12, 4, 10, 11, 9, 7, 6, 16, 24, 24]
index = [0, 9, 14, 23, 29, 35, 38, 41, 47, 53, 61, 65, 71, 77, 89, 93, 103, 114, 123, 130, 136, 152, 176]


def train_epoch(model, data_loader, loss_criterion, optimizer, scheduler, epoch, device, logger, args, momentum_model):
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    model.train()
    acc_lv1_meter, acc_lv2_meter = utils.AverageMeter(), utils.AverageMeter()
    fgm = FGM(model)
    pgd = PGD(model)
    loss_lv1, loss_lv2, kl_loss, distill_loss, loss_contrastive = 0, 0, 0, 0, 0
    start_time = time.time()
    num_total_steps = len(data_loader)
    for step, batch in enumerate(data_loader):
        frame_feat, frame_feat_mask = batch[0].to(device), batch[1].to(device)
        text_input_ids, text_segment_ids, text_attention_mask = batch[2].to(device), batch[3].to(device), batch[4].to(
            device)
        label1, label2 = batch[6].to(device), batch[7].to(device)
        tfidf = batch[8].to(device)
        total_loss = 0

        if args.use_rdrop:
            batch_size = len(frame_feat)
            double_frame_feat = torch.cat([frame_feat, frame_feat], dim=0)
            double_frame_feat_mask = torch.cat([frame_feat_mask, frame_feat_mask], dim=0)
            double_text_input_ids = torch.cat([text_input_ids, text_input_ids], dim=0)
            double_text_segment_ids = torch.cat([text_segment_ids, text_segment_ids], dim=0)
            double_text_attention_mask = torch.cat([text_attention_mask, text_attention_mask], dim=0)
            outputs = model(double_frame_feat, double_frame_feat_mask, double_text_input_ids, double_text_segment_ids, double_text_attention_mask)
            output_lv2 = outputs[0]
            loss_lv2 = loss_criterion(output_lv2, torch.cat([label2, label2], dim=0))
            kl_loss = compute_kl_loss(output_lv2[:batch_size, :], output_lv2[batch_size:, :])
            total_loss += loss_lv2 + kl_loss

            output_lv2 = output_lv2[:batch_size]
        else:
            outputs = model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask)
            output_lv2 = outputs[0]

            if 'binary' in args.loss:
                onehot_labels = torch.zeros((label2.size()[0], 200), device=device)
                onehot_labels.scatter_(1, label2.unsqueeze(-1), 1)
                loss_lv2 = loss_criterion(output_lv2, onehot_labels)
            else:
                loss_lv2 = loss_criterion(output_lv2, label2)
            total_loss += loss_lv2

        if args.use_contrastive:
            loss_contrastive = outputs[2]
            total_loss += loss_contrastive

        if args.use_lv1:
            output_lv1 = outputs[1]
            loss_lv1 = loss_criterion(output_lv1, label1)
            total_loss += loss_lv1

        if args.use_ema or args.use_distill:
            with torch.no_grad():
                utils.momentum_update([model, momentum_model], args.momentum)

        if args.use_distill:
            if epoch > 0:
                alpha = args.alpha
            else:
                alpha = args.alpha * min(1.0, step / len(data_loader))
            # alpha = args.alpha
            with torch.no_grad():
                output_lv2_m = momentum_model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask)
                logits_lv2_m = torch.softmax(output_lv2_m, dim=1)

            distill_loss = torch.mean(torch.sum(-F.log_softmax(output_lv2, dim=1) * logits_lv2_m, dim=1)) * alpha
            total_loss += distill_loss

        total_loss /= args.accumu_grad_step
        total_loss.backward()

        if args.use_fgm:
            fgm.attack(emb_name='word_embeddings')
            output_adv = model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask)[0]
            loss_adv = loss_criterion(output_adv, label2)
            loss_adv /= args.accumu_grad_step
            loss_adv.backward()
            fgm.restore(emb_name='word_embeddings')
        elif args.use_pgd:
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(emb_name='word_embeddings', is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                output_adv = model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask)[0]
                loss_adv = loss_criterion(output_adv, label2)
                loss_adv /= args.accumu_grad_step
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore(emb_name='word_embeddings')  # 恢复embedding参数

        #         if args.rank == 0:
        #             for n, p in model.named_parameters():
        #                 if p.requires_grad and p.grad is None:
        #                     print(n)

        if (step + 1) % args.accumu_grad_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        logits = F.softmax(output_lv2, dim=1).detach().cpu().numpy()
        pred_label2 = np.argmax(logits, axis=1)
        pred_label1 = [lv2id_to_lv1id(l) for l in pred_label2]
        label2 = label2.cpu().numpy()
        label1 = label1.cpu().numpy()

        micro_f1_label2 = f1_score(label2, pred_label2, average='micro')
        macro_f1_label2 = f1_score(label2, pred_label2, average='macro')
        micro_f1_label1 = f1_score(label1, pred_label1, average='micro')
        macro_f1_label1 = f1_score(label1, pred_label1, average='macro')

        acc_lv1 = accuracy_score(label1, pred_label1)
        acc_lv2 = accuracy_score(label2, pred_label2)
        acc_lv1_meter.update(acc_lv1, len(label1))
        acc_lv2_meter.update(acc_lv2, len(label2))

        f1 = (micro_f1_label2 + macro_f1_label2 + micro_f1_label1 + macro_f1_label1) / 4

        if (step + 1) % (200) == 0 and args.rank == 0:
            time_per_step = (time.time() - start_time) / max(1, step)
            remaining_time = time_per_step * (num_total_steps - step)
            remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            print(
                'train epoch: {} step: {} eta: {}| loss: {} loss_lv1: {} loss_lv2: {} kl_loss: {} distill_loss: {} '
                'contras_loss: {} | mean_f1 score: {}, lv1_micro_f1: {}, lv1_macro_f1: {},'
                ' lv2_micro_f1: {}, lv2_macro_f1: {} lv1_acc: {}, lv2_acc: {}'.format(epoch, step, remaining_time,
                                                                                      total_loss,
                                                                                      loss_lv1, loss_lv2,
                                                                                      kl_loss, distill_loss,
                                                                                      loss_contrastive,
                                                                                      round(f1, 6),
                                                                                      round(micro_f1_label1, 6),
                                                                                      round(macro_f1_label1, 6),
                                                                                      round(micro_f1_label2, 6),
                                                                                      round(macro_f1_label2, 6),
                                                                                      round(acc_lv1_meter.avg, 6),
                                                                                      round(acc_lv2_meter.avg, 6),
                                                                                      ))


def val_epoch(model, data_loader, loss_criterion, epoch, device, logger, args):
    model.eval()
    all_label1, all_label2 = [], []
    all_pred1, all_pred2 = [], []
    all_sample_ids = []
    # all_title, all_asr, all_ocr = [], [], []

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            frame_feat, frame_feat_mask = batch[0].to(device), batch[1].to(device)
            text_input_ids, text_segment_ids, text_attention_mask = batch[2].to(device), batch[3].to(device), batch[
                4].to(device)
            label1, label2 = batch[6].to(device), batch[7].to(device)
            tfidf = batch[8].to(device)

            output = model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask)
            if 'binary' in args.loss:
                onehot_labels = torch.zeros((label2.size()[0], 200), device=device)
                onehot_labels.scatter_(1, label2.unsqueeze(-1), 1)
                loss = loss_criterion(output, onehot_labels)
            else:
                loss = loss_criterion(output, label2)
            logits = F.softmax(output, dim=1)
            pred_label2 = torch.argmax(logits, dim=1)

            all_pred2.append(pred_label2)
            all_label2.append(label2)
            all_label1.append(label1)
            all_sample_ids.append(batch[5])

    all_pred2 = torch.cat(all_pred2, dim=0)
    all_label1 = torch.cat(all_label1, dim=0)
    all_label2 = torch.cat(all_label2, dim=0)

    if args.distributed:
        pred2_tensor_list = [torch.zeros_like(all_pred2) for _ in range(dist.get_world_size())]
        label1_tensor_list = [torch.zeros_like(all_label1) for _ in range(dist.get_world_size())]
        label2_tensor_list = [torch.zeros_like(all_label2) for _ in range(dist.get_world_size())]
        dist.all_gather(pred2_tensor_list, all_pred2)
        dist.all_gather(label1_tensor_list, all_label1)
        dist.all_gather(label2_tensor_list, all_label2)
        all_pred2 = torch.cat(pred2_tensor_list, dim=0)
        all_label1 = torch.cat(label1_tensor_list, dim=0)
        all_label2 = torch.cat(label2_tensor_list, dim=0)

    all_pred2 = all_pred2.detach().cpu().numpy()
    all_label1 = all_label1.detach().cpu().numpy()
    all_label2 = all_label2.detach().cpu().numpy()
    all_pred1 = [lv2id_to_lv1id(l) for l in all_pred2]

    micro_f1_label2 = f1_score(all_label2, all_pred2, average='micro')
    macro_f1_label2 = f1_score(all_label2, all_pred2, average='macro')
    micro_f1_label1 = f1_score(all_label1, all_pred1, average='micro')
    macro_f1_label1 = f1_score(all_label1, all_pred1, average='macro')

    acc_lv1 = accuracy_score(all_label1, all_pred1)
    acc_lv2 = accuracy_score(all_label2, all_pred2)

    f1 = (micro_f1_label2 + macro_f1_label2 + micro_f1_label1 + macro_f1_label1) / 4

    # with open('./src/xdl_src/output/val_ids_{}.txt'.format(args.rank), 'w') as fw:
    #     for id in all_sample_ids:
    #         fw.write('{}\n'.format(id))

    if args.rank == 0:
        print(all_pred2.shape)
        logger.info('val epoch: {} | loss: {}, mean_f1 score: {}, lv1_micro_f1: {}, lv1_macro_f1: {},'
                    ' lv2_micro_f1: {}, lv2_macro_f1: {}, lv1_acc: {}, lv2_acc: {}\n'.format(epoch, loss, round(f1, 6),
                                                                                             round(micro_f1_label1, 6),
                                                                                             round(macro_f1_label1, 6),
                                                                                             round(micro_f1_label2, 6),
                                                                                             round(macro_f1_label2, 6),
                                                                                             round(acc_lv1, 6),
                                                                                             round(acc_lv2, 6),
                                                                                             ))


def prediction(model, data_loader, device):
    model.eval()
    all_pred2 = []
    all_pred2_logits = []
    all_sample_ids = []

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            frame_feat, frame_feat_mask, title, asr, ocr = batch[0].to(device), batch[1].to(device), batch[2], batch[3], \
                                                           batch[4]
            sample_id = batch[5]
            tfidf = batch[8].to(device)

            output = model(frame_feat, frame_feat_mask, title, asr, ocr, tfidf)[0]
            logits = output
            pre_label2 = torch.argmax(logits, dim=1)

            # all_pred2 += [lv2id_to_category_id(p) for p in pre_label2]
            all_pred2.append(pre_label2)
            # all_sample_ids += sample_id
            all_pred2_logits.append(logits)

    all_pred2 = torch.cat(all_pred2, dim=0)
    all_pred2_logits = torch.cat(all_pred2_logits, dim=0)

    if args.distributed:
        pred2_tensor_list = [torch.zeros_like(all_pred2) for _ in range(dist.get_world_size())]
        pred2_logits_tensor_list = [torch.zeros_like(all_pred2_logits) for _ in range(dist.get_world_size())]
        dist.all_gather(pred2_tensor_list, all_pred2)
        dist.all_gather(pred2_logits_tensor_list, all_pred2_logits)
        all_pred2 = torch.cat(pred2_tensor_list, dim=0)
        all_pred2_logits = torch.cat(pred2_logits_tensor_list, dim=0)

    print(all_pred2.shape)
    all_pred2 = all_pred2.cpu().numpy()
    # all_pred2 = [lv2id_to_category_id(l) for l in all_pred2]

    if args.rank == 0:
        with open('./src/xdl_src/output/{}.csv'.format(args.flag), 'w') as fw:
            for i in range(len(all_pred2)):
                fw.write('{},{}\n'.format(i, lv2id_to_category_id(all_pred2[i])))

    # all_pred2_logits = np.concatenate(all_pred2_logits, axis=0).tolist()
    # with open('./src/xdl_src/output/{}_logits.jsonl'.format(args.flag), 'w') as fw:
    #     for i in range(len(all_sample_ids)):
    #         fw.write('{}\n'.format(json.dumps({'id': all_sample_ids[i], 'logits': all_pred2_logits[i]})))


def main(args):
    # init
    if args.distributed:
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    logger = None
    if args.rank == 0:
        print('model path: ', os.path.exists(args.model_save_path))
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
        logger = utils.create_logger(
            os.path.join(args.output_dir, 'log_{}_{}_{}_{}.txt'.format(args.flag, year, month, day)))

        logger.info(args)
    print(args.rank)

    start_epoch = 0
    device = torch.device(args.device)

    # create dataset and dataloader
    dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path,
                             anno_path=args.anno_path, use_raw_image=args.use_raw_image, max_frames=args.max_frames,
                             use_aug=args.use_aug, args=args)

    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    #     train_dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path, phase='train',
    #                              anno_path=args.train_anno_path, use_raw_image=args.use_raw_image, max_frames=args.max_frames,
    #                              use_aug=args.use_aug, args=args)
    #     val_dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path, phase='val',
    #                              anno_path=args.val_anno_path, use_raw_image=args.use_raw_image, max_frames=args.max_frames,
    #                              use_aug=False, args=args)

    print(len(train_dataset), len(val_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True,
                                  drop_last=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, pin_memory=True,
                                drop_last=False, num_workers=args.num_workers, prefetch_factor=args.prefetch)

    # create model
    config = {'text_encoder_arch': args.text_encoder_arch, 'text_encoder_path': args.text_encoder_path,
              'text_encoder_config_path': args.text_encoder_config_path,
              'visual_encoder_path': args.visual_encoder_path, 'visual_encoder_arch': args.visual_encoder_arch,
              'visual_encoder_config_path': args.visual_encoder_config_path,
              'use_visual_encoder': args.use_visual_encoder,
              'frame_encoder_path': args.frame_encoder_path, 'frame_encoder_arch': args.frame_encoder_arch,
              'frame_encoder_config_path': args.frame_encoder_config_path,

              'multimodal_config_path': args.multimodal_config_path, 'fusion': args.fusion,
              'cross_type': args.cross_type,
              'text_embed_dim': args.text_embed_dim, 'visual_embed_dim': args.visual_embed_dim,
              'mm_embed_dim': args.mm_embed_dim,
              'use_asr': args.use_asr, 'use_ocr': args.use_ocr, 'max_len': args.max_len,
              'max_title_len': args.max_title_len, 'max_asr_len': args.max_asr_len, 'max_ocr_len': args.max_ocr_len,
              'use_prompt': args.use_prompt, 'use_tfidf': args.use_tfidf, 'tfidf_dim': args.tfidf_dim,
              'label2_nums': 200, 'label1_nums': 23,
              'multitext_config_path': args.multitext_config_path,
              'use_momentum_text_encoder': args.use_momentum_text_encoder,
              'momentum': args.momentum, 'use_pooling': args.use_pooling, 'use_lv1': args.use_lv1,
              'asr_type': args.asr_type, 'ocr_type': args.ocr_type, 'use_contrastive': args.use_contrastive,
              'queue_size': args.queue_size, 'temp': args.temp, 'pooling': args.pooling,
              'truncation': args.truncation, 'spatial_dropout': args.spatial_dropout,
              'use_single_modal': args.use_single_modal, 'use_raw_image': args.use_raw_image, 'use_aug': args.use_aug,
              'frame_emb_type': args.frame_emb_type}
    if args.rank == 0:
        print(config)

    model = MultiModal_Classifier(config)
    model = model.to(device)
    model_without_ddp = model
    if (args.use_distill or args.use_ema) and args.phase == 'train':
        momentum_model = MultiModal_Classifier(config).to(device)
        utils.copy_params([model, momentum_model])
        momentum_model.eval()
    else:
        momentum_model = None

    if args.frozen_bert_layers > 0:
        model.frozen_bert_layers(args.frozen_bert_layers)

    pretrained_names = ['text_encoder']
    if args.use_raw_image:
        pretrained_names += ['frame_encoder']
    optimizer, scheduler = utils.create_optimizer(model, lr_pretrained=args.lr_pretrained, lr_random=args.lr_random,
                                                  no_decay_names=['bias', 'LayerNorm.weight'],
                                                  pretrained_names=pretrained_names,
                                                  warmup_steps=args.warmup_steps,
                                                  max_steps=args.epochs * len(train_dataloader),
                                                  schedule_type=args.schedule_type)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        msg = model.load_state_dict(state_dict, strict=False)
        if args.rank == 0:
            print('load checkpoint from %s', args.checkpoint)
            print('load checkpoint: ', msg)

    if args.use_swa and args.phase == 'val':
        swa_raw_model = copy.deepcopy(model)
        model = utils.swa(swa_raw_model, args.model_save_path, swa_start=1, swa_end=3)
        model_without_ddp = model
        save_obj = {
                        'model': model.state_dict(),
                    }
        torch.save(save_obj, os.path.join(args.model_save_path, 'model_swa.pth'))
        print('saved swa model.')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    ### loss function
    if args.loss == 'cross_entropy':
        loss_criterion = nn.CrossEntropyLoss()
    elif args.loss == 'label_smooth':
        loss_criterion = LabelSmoothingCrossEntropy()
    elif args.loss == 'asl':
        loss_criterion = ASLSingleLabel(gamma_neg=2, gamma_pos=0, eps=0.1)
    elif args.loss == 'binary_cross_entropy':
        loss_criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'binary_asl':
        loss_criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.1)
    else:
        raise ValueError

    # train and eval
    if args.phase == 'train':
        for epoch in range(start_epoch, args.epochs):
            train_epoch(model, train_dataloader, loss_criterion, optimizer, scheduler, epoch, device, logger, args,
                        momentum_model)
            if val_size > 0:
                val_epoch(model, val_dataloader, loss_criterion, epoch, device, logger, args)
                if args.use_distill or args.use_ema:
                    val_epoch(momentum_model, val_dataloader, loss_criterion, epoch, device, logger, args)

            if args.rank == 0:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    #                     'optimizer': optimizer.state_dict(),
                    #                     'scheduler': scheduler.state_dict(),
                    #                     'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.model_save_path, 'checkpoint_{}.pth'.format(epoch)))

                if args.use_ema:
                    save_obj = {
                        'model': momentum_model.state_dict(),
                    }
                    torch.save(save_obj, os.path.join(args.model_save_path, 'checkpoint_m_{}.pth'.format(epoch)))
            if epoch >= 4:
                break
    elif args.phase == 'val':
        start_time = time.time()
        val_epoch(model, val_dataloader, loss_criterion, 0, device, logger, args)
        end_time = time.time()

        print('cost time: ', end_time - start_time)


#     elif args.phase == 'test':
#         print('start predicting...')
#         test_dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path,
#                                       anno_path=args.test_anno_path, use_raw_image=args.use_raw_image, max_frames=args.max_frames)

#         train_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [size - val_size, val_size],
#                                                                    generator=torch.Generator().manual_seed(args.seed))
#         print(len(test_dataset))

#         if args.distributed:
#             test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=False)
#         else:
#             test_sampler = None

#         test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, pin_memory=True,
#                                      drop_last=False, shuffle=False, num_workers=args.num_workers,
#                                      prefetch_factor=args.prefetch)
#         start_time = time.time()
#         prediction(model, test_dataloader, device)
#         end_time = time.time()
#         print('cost time: ', end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output/logs')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/base')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--flag', type=str, default='')
    parser.add_argument('--accumu_grad_step', type=int, default=1)
    parser.add_argument('--anno_path', type=str, default='')
    parser.add_argument('--train_data_path', type=str, default='./data/wxchallenge_example_data')
    parser.add_argument('--train_anno_path', type=str, default='')
    parser.add_argument('--val_data_path', type=str, default='./data/wxchallenge_example_data')
    parser.add_argument('--val_anno_path', type=str, default='')
    parser.add_argument('--test_data_path', type=str, default='./data/wxchallenge_example_data')
    parser.add_argument('--test_anno_path', type=str, default='')
    parser.add_argument('--zip_frame_path', type=str, default='')
    parser.add_argument('--zip_feat_path', type=str, default='')
    parser.add_argument('--use_raw_image', action='store_true')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=32)

    parser.add_argument('--text_encoder_path', type=str, default='./pretrained_model/roberta/')
    parser.add_argument('--text_encoder_config_path', type=str, default='./pretrained_model/roberta/config.json')
    parser.add_argument('--text_encoder_arch', type=str, default='bert')
    parser.add_argument('--visual_encoder_path', type=str, default='./pretrained_model/vit_base/')
    parser.add_argument('--visual_encoder_arch', type=str, default='transformer')
    parser.add_argument('--visual_encoder_config_path', type=str, default='./configs/visual_encoder_config.json')
    parser.add_argument('--frame_encoder_path', type=str, default='./pretrained_model/vit_base/')
    parser.add_argument('--frame_encoder_arch', type=str, default='swin')
    parser.add_argument('--frame_encoder_config_path', type=str, default='./configs/visual_encoder_config.json')
    parser.add_argument('--frame_emb_type', type=str, default='frame')
    parser.add_argument('--multimodal_config_path', type=str, default='./configs/cross_attention_config.json')
    parser.add_argument('--multitext_config_path', type=str, default='./configs/cross_attention_config.json')

    parser.add_argument('--use_visual_encoder', action='store_true')
    parser.add_argument('--text_embed_dim', type=int, default=768)
    parser.add_argument('--visual_embed_dim', type=int, default=512)
    parser.add_argument('--mm_embed_dim', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_title_len', type=int, default=50)
    parser.add_argument('--max_asr_len', type=int, default=512)
    parser.add_argument('--max_ocr_len', type=int, default=100)
    parser.add_argument('--fusion', type=str, default='cross_attention')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--device_ids', default='0')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--cross_type', type=str, default='text')
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--model_type', type=str, default='v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument('--use_momentum_text_encoder', action='store_true')
    parser.add_argument('--use_asr', action='store_true')
    parser.add_argument('--use_ocr', action='store_true')
    parser.add_argument('--use_fgm', action='store_true')
    parser.add_argument('--use_pgd', action='store_true')
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--use_tfidf', action='store_true')
    parser.add_argument('--use_rdrop', action='store_true')
    parser.add_argument('--use_pooling', action='store_true')
    parser.add_argument('--use_distill', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--use_lv1', action='store_true')
    parser.add_argument('--use_single_modal', action='store_true')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--frozen_bert_layers', type=int, default=0)
    parser.add_argument('--tfidf_dim', type=int, default=256)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--lr_pretrained', type=float, default=1e-5)
    parser.add_argument('--lr_random', type=float, default=1e-4)
    parser.add_argument('--schedule_type', type=str, default='warmup')
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument('--asr_type', type=int, default=0)
    parser.add_argument('--ocr_type', type=int, default=0)
    parser.add_argument('--queue_size', type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--pooling', type=str, default='')
    parser.add_argument('--truncation', type=str, default='head')
    parser.add_argument("--spatial_dropout", type=float, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    main(args)