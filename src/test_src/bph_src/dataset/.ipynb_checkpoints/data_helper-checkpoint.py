import json
import random
import zipfile
from io import BytesIO
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from transformers import BertTokenizer
import os
import sys
from PIL import Image

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '../'))
from category_id_map import category_id_to_lv2id, category_id_to_lv1id_2


def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, args.train_zip_feats)
    train_index, val_index = [i for i in range(90000)], [i for i in range(90000, 100000)]

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
        
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_frame_dir: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        
        self.end2end = args.end2end
        
        
        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_frame_dir = zip_frame_dir
        self.zip_feat_path = zip_feats

        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        
        # we use the standard image transform as in the offifical Swin-Transformer.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 多线程
        self.handles = [None for _ in range(args.num_workers)]
        

    def __len__(self) -> int:
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
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
                # if num_frames // self.max_frame == 1:
                #     select_inds = list(range(0, num_frames))
                #     head_half_frames = (num_frames - self.max_frame) * 2
                #     tail_half_frames = self.max_frame - head_half_frames // 2
                #     select_inds = list(range(0, head_half_frames, 2)) + select_inds[-tail_half_frames:]
                # else:
                #     select_inds = list(range(0, num_frames, num_frames // self.max_frame))[:self.max_frame]
                
                
            else:
                # randomly sample when test mode is False
                # select_inds = list(range(num_frames))
                # random.shuffle(select_inds)
                # select_inds = select_inds[:self.max_frame]
                # select_inds = sorted(select_inds)
                
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
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
                # if num_frames // self.max_frame == 1:
                #     select_inds = list(range(0, num_frames))
                #     head_half_frames = (num_frames - self.max_frame) * 2
                #     tail_half_frames = self.max_frame - head_half_frames // 2
                #     select_inds = list(range(0, head_half_frames, 2)) + select_inds[-tail_half_frames:]
                # else:
                #     select_inds = list(range(0, num_frames, num_frames // self.max_frame))[:self.max_frame]
                
            else:
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
        # return frame.half(), mask.half()
        return frame.half(),mask.half()

    def tokenize_text(self, text: str, seq_length: int) -> tuple:
        # encoded_inputs = self.tokenizer(text, max_length=seq_length, padding='max_length', truncation=True)
        encoded_inputs = self.tokenizer.encode_plus(text, max_length=seq_length, padding='max_length', truncation=True)
        input_ids = np.array(encoded_inputs['input_ids'], dtype=np.int64)
        mask = np.array(encoded_inputs['attention_mask'], dtype=np.int64)
        token_type_ids = np.array(encoded_inputs['token_type_ids'], dtype=np.int64)

        return input_ids, mask, token_type_ids

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        if self.end2end:
            frame_input, frame_mask = self.get_visual_frames(idx)
        else:
            # 多线程
            worker_info = torch.utils.data.get_worker_info()
            frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load title tokens
        # 拼接title和asr
        title = self.anns[idx]['title']
        # print(title)
        asr = self.anns[idx]['asr']
        ocr_ = self.anns[idx]['ocr']
        ocr = ''
        for dic in ocr_:
            ocr += dic['text']

        title = self.tokenizer.tokenize(title)
        asr = self.tokenizer.tokenize(asr)
        ocr_text = self.tokenizer.tokenize(ocr)

        text = title + ['[SEP]'] + asr + ['[SEP]'] + ocr_text

        title_input, title_mask, token_type_ids = self.tokenize_text(text, self.bert_seq_length)
        # ocr_input, ocr_mask, ocr_token_type_ids = self.tokenize_text(ocr_text, 128)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask,
            token_type_ids=token_type_ids,
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])
            # lv1_label = category_id_to_lv1id_2(self.anns[idx]['category_id'][:2])
            # data['lv1_label'] = torch.LongTensor([lv1_label])
        return data
