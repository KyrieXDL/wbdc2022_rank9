import os
import torch
from mydataset.wechat_datatset import Wechat_Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import argparse
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
import time

def main(args):
    utils.fix_random_seeds(args.seed)

    # create dataset and dataloader
    dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path,
                             anno_path=args.anno_path, use_raw_image=args.use_raw_image, max_frames=args.max_frames,
                             use_aug=args.use_aug, args=args)

    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    print(len(train_dataset), len(val_dataset))
        
    val_ids = [val_dataset[i][5] for i in range(len(val_dataset))]
    
    with open(args.anno_path, 'r') as fr:
        data = json.loads(fr.readline())
        
    train_data, val_data = [], []
    for d in data:
        if d['id'] in val_ids:
            val_data.append(d)
        else:
            train_data.append(d)
            
    with open(os.path.join(args.output_dir, 'train_labeled.json'), 'w') as fw:
        fw.write(json.dumps(train_data, ensure_ascii=False))

    with open(os.path.join(args.output_dir, 'val_labeled.json'), 'w') as fw:
        fw.write(json.dumps(val_data, ensure_ascii=False))
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--zip_frame_path', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--zip_feat_path', type=str, default='/opt/ml/input/data/zip_feats_clip/labeled.zip')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/input/data/annotations')
    parser.add_argument('--use_raw_image', action='store_true')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--max_title_len', type=int, default=50)
    parser.add_argument('--max_asr_len', type=int, default=512)
    parser.add_argument('--max_ocr_len', type=int, default=100)
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--use_asr', action='store_true')
    parser.add_argument('--use_ocr', action='store_true')
    parser.add_argument('--asr_type', type=int, default=0)
    parser.add_argument('--ocr_type', type=int, default=0)
    parser.add_argument('--truncation', type=str, default='head')
    parser.add_argument('--text_encoder_path', type=str, default='./opensource_models/pretrain_models/chinese-macbert-base')

    args = parser.parse_args()

    main(args)