import os
import torch
from torch.utils.data import DataLoader
from mydataset.wechat_datatset_pretrain import Wechat_Dataset
from models.multimodal_pretrain import MultiModal_Pretrain
from models.multimodal_pretrain_v0 import MultiModal_Pretrain as MultiModal_Pretrain_v0
import utils
import argparse
from datetime import datetime
import torch.distributed as dist
import time
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast
import math


def train_epoch(model, data_loader, optimizer, scheduler, epoch, device, logger, args):
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    # model, optimizer = amp.initialize(model, optimizer, opt_level=“O1”)
    model.train()
    scaler = GradScaler()
    loss_mlm_meter, loss_itc_meter, loss_ima_meter, loss_itm_meter, loss_mfm_meter = [utils.AverageMeter() for _ in
                                                                                      range(5)]
    start_time = time.time()
    num_total_steps = len(data_loader)
    for step, batch in enumerate(data_loader):
        frame_feat, frame_feat_mask = batch[0].to(device), batch[1].to(device)
        text_input_ids, text_segment_ids, text_attention_mask = batch[2].to(device), batch[3].to(device), batch[4].to(
            device)

        if epoch > 0:
            alpha = args.alpha
        else:
            alpha = args.alpha * min(1.0, step / len(data_loader))

        if args.use_fp16:
            with autocast():
                output = model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask,
                               alpha=alpha)
        else:
            output = model(frame_feat, frame_feat_mask, text_input_ids, text_segment_ids, text_attention_mask,
                           alpha=alpha)

        loss_itc, loss_ima, loss_itm, loss_mlm, loss_mfm = output
        loss_itc, loss_ima, loss_itm, loss_mlm = loss_itc * args.itc_weight, loss_ima * args.ima_weight, loss_itm * args.itm_weight, loss_mlm * args.mlm_weight
        loss = loss_itc + loss_ima + loss_itm + loss_mlm + loss_mfm
        loss /= args.accumu_grad_step
        # loss.backward()

        # if args.rank == 0:
        #     for n, p in model.named_parameters():
        #         if p.requires_grad and p.grad is None:
        #             print(n)

        if (step + 1) % args.accumu_grad_step == 0:
            if args.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if 'itc' in args.tasks and not math.isnan(loss_itc.item()):
            loss_itc_meter.update(loss_itc.item(), len(frame_feat))
        if 'ima' in args.tasks and not math.isnan(loss_ima.item()):
            loss_ima_meter.update(loss_ima.item(), len(frame_feat))
        if 'itm' in args.tasks and not math.isnan(loss_itm.item()):
            loss_itm_meter.update(loss_itm.item(), len(frame_feat))
        if 'mlm' in args.tasks and not math.isnan(loss_mlm.item()):
            loss_mlm_meter.update(loss_mlm.item(), len(frame_feat))
        if 'mfm' in args.tasks and not math.isnan(loss_mfm.item()):
            loss_mfm_meter.update(loss_mfm.item(), len(frame_feat))

        if (step + 1) % (200) == 0 and args.rank == 0:
            time_per_step = (time.time() - start_time) / max(1, step)
            remaining_time = time_per_step * (num_total_steps - step)
            remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            lr = optimizer.param_groups[0]["lr"]
            # print(loss_itc, loss_ima, loss_itm, loss_mlm, loss_mfm)
            logger.info(
                'train epoch: {} step: {} lr: {} eta: {} | itc_loss: {} loss_ima: {} itm_loss: {} mlm_loss: {} mfm_loss: {}'.format(
                    epoch,
                    step,
                    round(lr, 8),
                    remaining_time,
                    round(loss_itc_meter.avg, 6),
                    round(loss_ima_meter.avg, 6),
                    round(loss_itm_meter.avg, 6),
                    round(loss_mlm_meter.avg, 6),
                    round(loss_mfm_meter.avg, 6)
                ))


def main(args):
    # init
    if args.distributed:
        utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    logger = None
    if args.rank == 0:
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
        logger = utils.create_logger(
            os.path.join(args.output_dir, 'log_{}_{}_{}_{}.txt'.format(args.flag, year, month, day)))

        print(args)
    print(args.rank)

    start_epoch = 0
    device = torch.device(args.device)

    # create dataset and dataloader
    train_dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path,
                                   anno_path=args.anno_path, use_raw_image=args.use_raw_image,
                                   labeled_zip_feat_path=args.labeled_zip_feat_path,
                                   labeled_anno_path=args.labeled_anno_path,
                                   max_frames=args.max_frames, num_workers=4,
                                   use_aug=False, args=args)

    print(len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True,
                                  drop_last=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)

    # create model
    config = {'text_encoder_arch': args.text_encoder_arch, 'text_encoder_path': args.text_encoder_path,
              'text_encoder_config_path': args.text_encoder_config_path,
              'visual_encoder_path': args.visual_encoder_path, 'visual_encoder_arch': args.visual_encoder_arch,
              'visual_encoder_config_path': args.visual_encoder_config_path,
              'use_visual_encoder': args.use_visual_encoder,
              'frame_encoder_path': args.frame_encoder_path, 'frame_encoder_arch': args.frame_encoder_arch,
              'frame_encoder_config_path': args.frame_encoder_config_path,

              'mlm_probability': args.mlm_probability, 'mfm_probability': args.mfm_probability,

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
              'momentum': args.momentum, 'asr_type': args.asr_type, 'ocr_type': args.ocr_type,
              'queue_size': args.queue_size, 'temp': args.temp, 'pooling': args.pooling,
              'truncation': args.truncation, 'spatial_dropout': args.spatial_dropout, 'use_hard_negs': args.use_hard_negs,
              'use_raw_image': args.use_raw_image, 'frame_emb_type': args.frame_emb_type, 'tasks': args.tasks}
    if args.rank == 0:
        print(config)
    
    if args.model_type == 0:
        model = MultiModal_Pretrain_v0(config)
    else:
        model = MultiModal_Pretrain(config)

    model = model.to(device)
    model_without_ddp = model

    # if args.frozen_bert_layers > 0:
    #     model.frozen_bert_layers(args.frozen_bert_layers)

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
            checkpoint = torch.load(os.path.join(args.model_save_path, 'optimizer.pth'), map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print('resume...')

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s', args.checkpoint)
        print('load checkpoint: ', msg)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # train
    for epoch in range(start_epoch, args.epochs):
        train_epoch(model, train_dataloader, optimizer, scheduler, epoch, device, logger, args)

        if args.rank == 0:
            save_obj = {
                'model': model_without_ddp.state_dict(),
            }
            optimizer_obj = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.model_save_path, 'checkpoint_{}.pth'.format(epoch)))
            torch.save(optimizer_obj, os.path.join(args.model_save_path, 'optimizer.pth'))
        
        if epoch >= args.max_epochs:
            break
        # dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output/logs')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/base')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--flag', type=str, default='')
    parser.add_argument('--accumu_grad_step', type=int, default=1)
    parser.add_argument('--train_data_path', type=str, default='./data/wxchallenge_example_data')
    parser.add_argument('--anno_path', type=str, default='')
    parser.add_argument('--zip_frame_path', type=str, default='')
    parser.add_argument('--zip_feat_path', type=str, default='')
    parser.add_argument('--labeled_anno_path', type=str, default='')
    parser.add_argument('--labeled_zip_feat_path', type=str, default='')
    parser.add_argument('--use_raw_image', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--truncation', type=str, default='head')
    parser.add_argument('--asr_type', type=int, default=0)
    parser.add_argument('--ocr_type', type=int, default=0)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--use_labeled_data', action='store_true')

    parser.add_argument('--text_encoder_path', type=str, default='./pretrained_model/roberta/')
    parser.add_argument('--text_encoder_config_path', type=str, default='./pretrained_model/roberta/config.json')
    parser.add_argument('--text_encoder_arch', type=str, default='bert')
    parser.add_argument('--visual_encoder_arch', type=str, default='transformer')
    parser.add_argument('--visual_encoder_config_path', type=str, default='./configs/visual_encoder_config.json')
    parser.add_argument('--visual_encoder_path', type=str, default='./configs/visual_encoder_config.json')
    parser.add_argument('--frame_encoder_path', type=str, default='./pretrained_model/vit_base/')
    parser.add_argument('--frame_encoder_arch', type=str, default='swin')
    parser.add_argument('--frame_encoder_config_path', type=str, default='./configs/visual_encoder_config.json')
    parser.add_argument('--frame_emb_type', type=str, default='frame')
    parser.add_argument('--multimodal_config_path', type=str, default='./configs/cross_attention_config.json')
    parser.add_argument('--multitext_config_path', type=str, default='./configs/cross_attention_config.json')
    parser.add_argument('--model_type', type=int, default=0)

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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument('--use_momentum_text_encoder', action='store_true')
    parser.add_argument('--use_asr', action='store_true')
    parser.add_argument('--use_ocr', action='store_true')
    parser.add_argument('--use_fgm', action='store_true')
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--use_tfidf', action='store_true')
    parser.add_argument('--use_rdrop', action='store_true')
    parser.add_argument('--use_pooling', action='store_true')
    parser.add_argument('--use_hard_negs', action='store_true')
    parser.add_argument('--frozen_bert_layers', type=int, default=0)
    parser.add_argument('--tfidf_dim', type=int, default=256)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    parser.add_argument('--lr_pretrained', type=float, default=1e-5)
    parser.add_argument('--lr_random', type=float, default=1e-4)
    parser.add_argument('--queue_size', type=int, default=256)
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--mfm_probability', type=float, default=0.25)
    parser.add_argument('--title_mlm_probability', type=float, default=0.15)
    parser.add_argument('--asr_mlm_probability', type=float, default=0.15)
    parser.add_argument('--ocr_mlm_probability', type=float, default=0.15)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--schedule_type', type=str, default='warmup')
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--itc_weight", type=float, default=1)
    parser.add_argument("--ima_weight", type=float, default=1)
    parser.add_argument("--itm_weight", type=float, default=1)
    parser.add_argument("--mlm_weight", type=float, default=1)
    parser.add_argument('--tasks', type=str, default='mlm,itm,itc')
    parser.add_argument('--pooling', type=str, default='')
    parser.add_argument("--spatial_dropout", type=float, default=0)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    main(args)
