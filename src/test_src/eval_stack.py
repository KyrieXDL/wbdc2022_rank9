import os
import sys
import torch
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
import time
import copy
import numpy as np
import pickle
import time
from functools import partial

base_dir = os.path.dirname(__file__)

sys.path.append(os.path.join(base_dir, './xdl_src'))
from mydataset.wechat_datatset import Wechat_Dataset
from models.multimodal_classifiser import MultiModal_Classifier
from models.model_stack import Model_Stack
from mydataset.category_id_map import category_id_to_lv2id, category_id_to_lv1id, lv2id_to_category_id, lv2id_to_lv1id

sys.path.append(os.path.join(base_dir, './bph_src'))
from models.model import MultiModal
from models.model_large import MultiModalLarge

def prediction(model_stack, data_loader, device):
    model_stack.eval()
    all_pred2 = []
    all_pred2_logits = []
    all_sample_ids = []
    real_labels = []

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            frame_feat, frame_feat_mask = batch[0].to(device), batch[1].to(device)
            text_input_ids, text_segment_ids, text_attention_mask = batch[2].to(device), batch[3].to(device), batch[4].to(device)
            sample_id = batch[5]
            label = batch[7]
            tfidf = batch[8].to(device)
            
            output_list = model_stack(frame_feat, frame_feat_mask, text_input_ids, text_attention_mask, text_segment_ids)
            output1, output2, output3, output4 = output_list
            logits = (0.45 * output1 + 0.55 * output2) * 0.7 + (0.7 * output3 + output4 * 0.3) * 0.3
            
            # output1, output2 = output_list
            # logits = 0.45 * output1 + 0.55 * output2
            
            pre_label2 = torch.argmax(logits, dim=1)

            all_pred2.append(pre_label2)
            real_labels.append(label)
            all_sample_ids += sample_id
            all_pred2_logits.append(logits)

    all_pred2 = torch.cat(all_pred2, dim=0)
    real_labels = torch.cat(real_labels, dim=0)
    
    all_pred2_logits = torch.cat(all_pred2_logits, dim=0)

    all_pred2 = all_pred2.cpu().numpy()
    all_pred2 = [lv2id_to_category_id(l) for l in all_pred2]
    
    real_labels = real_labels.cpu().numpy()
    real_labels = [lv2id_to_category_id(l) for l in real_labels]
    

    with open('./result1.csv', 'w') as fw:
        for i in range(len(all_pred2)):
            fw.write('{},{}\n'.format(all_sample_ids[i], all_pred2[i], real_labels[i]))

            
def load_model1(args, config):
    model = MultiModal_Classifier(config)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s', args.checkpoint)
        print('load checkpoint: ', msg)
    return model


def load_model2(args):
    # bph
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(msg)
    return model

def load_model3(args):
    model = MultiModalLarge(args)
    checkpoint = torch.load(args.ckpt_file2, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(msg)
    return model
         
    
def main(args):
    device = torch.device(args.device)

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
              'use_single_modal': args.use_single_modal, 'use_raw_image': False, 'use_aug': args.use_aug,
              'frame_emb_type': args.frame_emb_type}
    if args.rank == 0:
        print(config)

    model1 = load_model1(args, config)
    args.checkpoint = args.checkpoint2
    model2 = load_model1(args, config)
    # bph
    print(args.bert_dir)
    model3 = load_model2(args)
    args.bert_dir = args.bert_dir2
    print(args.bert_dir)
    # large
    model4 = load_model3(args)
    
    # model_stack = Model_Stack([model1, model2], config)
    model_stack = Model_Stack([model1, model2, model3, model4], config)
    

    if torch.cuda.is_available():
        model_stack.half()
        model_stack = torch.nn.parallel.DataParallel(model_stack.cuda())

    print('start predicting...')
    start_time = time.time()
    test_dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path,
                                  anno_path=args.test_anno_path, use_raw_image=args.use_raw_image, 
                                  max_frames=args.max_frames,
                                  args=args)
    val_index = [i for i in range(90000, 90100)]
    test_dataset = torch.utils.data.Subset(test_dataset, val_index)
    print(len(test_dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                                 shuffle=False, num_workers=args.num_workers,
                                 prefetch_factor=args.prefetch)
    prediction(model_stack, test_dataloader, device)
    end_time = time.time()
    print('cost time: ', end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output/logs')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/base')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--checkpoint2', type=str, default='')
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
    
    
    
    ###########################
    root_path = os.path.join(base_dir, '../')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')


    # ========================= Data Configs ==========================
    #  /opt/ml/input/data
    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/test.json')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/test/')
    parser.add_argument('--test_zip_feats', type=str, default='/opt/ml/wxcode/zip_feats/test.zip')
    
    parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for testing duration per worker")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--ckpt_file', type=str, default='data/save/v1/model_.bin')
    parser.add_argument('--ckpt_file2', type=str, default='data/save/v1/model_.bin')
    
    # ========================= Learning Configs ==========================
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    
    # ========================== Swin ===================================
    parser.add_argument('--backbone', 
                        type=str,
                        default='swin-tiny',help='choices of backbone:efficientnet,van,swin-tiny,convnext')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default=root_path + 'data/pretrain_models/chinese-macbert-base')
    parser.add_argument('--bert_dir2', type=str, default=root_path + 'data/pretrain_models/chinese-macbert-large')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")
    
    
    parser.add_argument('--input_shape', default=[224, 224], help='shape of imgs')
    # 端到端 end to end
    parser.add_argument('--end2end', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fusion_layer', type=int, default=18)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids

    main(args)
