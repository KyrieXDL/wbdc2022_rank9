# encoding=utf-8
import json
import zipfile
import random
from io import BytesIO
import os
import numpy as np
import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor


# 记得修改为多线程
class WXDataset(torch.utils.data.Dataset):
    def __init__(self, args,
                 ann_path: str,
                 zip_frame_dir: str,
                 zip_feats: str):
        self.output_dict = {}
        self.end2end = args.end2end
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True)
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.zip_frame_dir = zip_frame_dir
        self.zip_feat_path = zip_feats
        
        # load train annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        
        # self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # 多线程
        self.handles = [None for _ in range(args.num_workers)]
        
        # we use the standard image transform as in the offifical Swin-Transformer.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.anns)
    
    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        # 多线程
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)

        # 不多线程
        # raw_feats = np.load(BytesIO(self.handles.read(name=f'{vid}.npy')), allow_pickle=True)
        
        #---------------------------------------------------#
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        #---------------------------------------------------#
        num_frames, feat_dim = raw_feats.shape
        
        #---------------------------------------------------## if occur error,change 16 to 32
        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)#32->16
        mask = np.ones((self.max_frame,), dtype=np.int32)#32->16
        #---------------------------------------------------#
        
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            step = num_frames // self.max_frame
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frame]
            
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask
    
    
    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        
        #---------------------------------------------------#32->16 if occur error,change 16 to 32
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.int32)
        #---------------------------------------------------#
        
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
           
            # randomly sample when test mode is False
            # select_inds = list(range(num_frames))
            # random.shuffle(select_inds)
            # select_inds = select_inds[:self.max_frame]
            # select_inds = sorted(select_inds)
            step = num_frames // self.max_frame
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frame]
        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def tokenize_text(self, text: str, seq_length: int) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=seq_length, padding='max_length', truncation=True)
        # encoded_inputs = self.tokenizer.encode_plus(text, max_length=seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'], dtype=np.int64)
        mask = np.array(encoded_inputs['attention_mask'], dtype=np.int64)
        token_type_ids = np.array(encoded_inputs['token_type_ids'], dtype=np.int64)

        return input_ids, mask, token_type_ids

    def __getitem__(self, index):

        # Step 1, load visual features from zipfile.
        if self.end2end:
            frame_input, frame_mask = self.get_visual_frames(index)
        else:
            worker_info = torch.utils.data.get_worker_info()
            frame_input, frame_mask = self.get_visual_feats(worker_info.id, index)
            # frame_input, frame_mask = self.get_visual_feats(0, index)

        # 拼接title和asr
        title = self.anns[index]['title']
        asr = self.anns[index]['asr']
        ocr_ = self.anns[index]['ocr']
        ocr = ''
        for dic in ocr_:
            ocr += dic['text']
        text = title + '[SEP]' + asr + '[SEP]' + ocr

        text_input, text_mask, token_type_ids = self.tokenize_text(text, self.bert_seq_length)
        o = {}
        o['frame_input'] = frame_input
        o['frame_mask'] = frame_mask

        o['text_input'] = text_input
        o['mask'] = text_mask
        return o
