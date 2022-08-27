from torch.utils.data import Dataset, DataLoader
import torch
import json
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
import random
import re
from mydataset.category_id_map import category_id_to_lv2id, category_id_to_lv1id
import os
import zipfile
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from mydataset.randaugment import RandomAugment
from transformers.models.bert import BertModel, BertTokenizer, BertConfig


class Wechat_Dataset(Dataset):
    def __init__(self, zip_frame_path='', zip_feat_path='', anno_path='', phase='train', use_raw_image=False,
                 num_workers=1, max_frames=32, data_size=-1, use_aug=False, args=None):
        super(Wechat_Dataset, self).__init__()
        self.phase = phase
        self.max_frame = max_frames
        self.zip_frame_dir = zip_frame_path
        self.zip_feat_path = zip_feat_path
        self.handles = [None for _ in range(num_workers)]
        self.use_raw_image = use_raw_image
        self.use_aug = use_aug
        self.max_title_len = args.max_title_len
        self.max_asr_len = args.max_asr_len
        self.max_ocr_len = args.max_ocr_len
        self.max_len = min(self.max_title_len + self.max_asr_len + self.max_ocr_len + 4, args.max_len)
        self.use_prompt, self.use_asr, self.use_ocr = args.use_prompt, args.use_asr, args.use_ocr
        self.truncation = args.truncation
        self.asr_type, self.ocr_type = args.asr_type, args.ocr_type

        with open(anno_path, 'r', encoding='utf8') as f:
            anns = json.load(f)
        #     random.shuffle(anns)
        # data_size = len(anns) if data_size <= 0 else data_size
        # self.data = anns[:data_size] if phase in ('train', 'test') else anns[-data_size:]
        self.data = anns
        self.transform = self.build_trans()
        self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder_path)

    def build_trans(self):
        if self.use_aug and self.phase == 'train':
            # normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            trans = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        return trans

    def __len__(self):
        return len(self.data)

    def get_visual_frames(self, idx):
        # read data from zipfile
        vid = self.data[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame,), dtype=torch.long)
        # select_inds = list(range(num_frames))

        if num_frames > self.max_frame:
            if num_frames // self.max_frame == 1:
                select_inds = list(range(0, num_frames))

                head_half_frames = (num_frames - self.max_frame) * 2
                tail_half_frames = self.max_frame - head_half_frames//2
                select_inds = list(range(0, head_half_frames, 2)) + select_inds[-tail_half_frames:]
            else:
                select_inds = list(range(0, num_frames, num_frames // self.max_frame))[:self.max_frame]
        else:
            select_inds = list(range(num_frames))

        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def get_visual_feats(self, worker_id, idx):
        # read data from zipfile
        vid = self.data[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)

        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0

        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def __getitem__(self, index):
        # raw_feats = self.data[index]['feat']
        title = self.data[index]['title']
        asr = self.data[index]['asr']
        ocr = self.data[index]['ocr']
        sample_id = self.data[index]['id']

        tfidf = torch.FloatTensor([0])
        if 'tfidf' in self.data[index]:
            tfidf = torch.FloatTensor(self.data[index]['tfidf'])
        ocr_text = ','.join([ocr[i]['text'] for i in range(len(ocr))])

        if self.use_raw_image:
            frame_input, frame_mask = self.get_visual_frames(index)
        else:
            frame_input, frame_mask = self.get_visual_feats(0, index)

        text_input_ids, text_segment_ids, text_attention_mask = self.process_text(title, asr, ocr_text)

        if 'category_id' in self.data[index]:
            label = self.data[index]['category_id']
            label1, label2 = category_id_to_lv1id(label), category_id_to_lv2id(label)
            return frame_input, frame_mask, text_input_ids, text_segment_ids, text_attention_mask, \
                   sample_id, label1, label2, tfidf

        return frame_input, frame_mask, text_input_ids, text_segment_ids, text_attention_mask\
            , sample_id, random.choice(
            list(range(0, 20))), random.choice(
            list(range(0, 99))), tfidf

    def process_text(self, title_text, asr_text=None, ocr_text=None):
        def filter_tokens(tokens):
            filtered_tokens = [tokens[0]] if len(tokens) > 0 else []
            cnt = 1
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i - 1]:
                    if cnt <= 2:
                        cnt += 1
                        filtered_tokens.append(tokens[i])
                else:
                    filtered_tokens.append(tokens[i])
                    cnt = 1
            return filtered_tokens

        if self.use_prompt:
            title_text = '标题文本：' + title_text
            asr_text = '语音文本：' + asr_text
            ocr_text = '图像文本：' + ocr_text

        # title
        title_tokens = self.tokenizer.tokenize(title_text)
        title_token_ids = self.tokenizer.convert_tokens_to_ids(title_tokens)
        title_len = min(self.max_len - 3, len(title_token_ids), self.max_title_len)
        if self.truncation == 'head_tail' and len(title_token_ids) > title_len:
            half1, half2 = title_len // 2, title_len - title_len // 2
            title_token_ids = title_token_ids[:half1] + title_token_ids[-half2:]
        else:
            title_token_ids = title_token_ids[:title_len]

        # asr
        asr_tokens = self.tokenizer.tokenize(asr_text)
        asr_tokens = filter_tokens(asr_tokens)
        asr_token_ids = self.tokenizer.convert_tokens_to_ids(asr_tokens)
        asr_len = min(self.max_len - 3 - title_len, len(asr_tokens), self.max_asr_len)
        if self.truncation == 'head_tail' and len(asr_token_ids) > asr_len:
            half1, half2 = asr_len // 2, asr_len - asr_len // 2
            asr_token_ids = asr_token_ids[:half1] + asr_token_ids[-half2:]
        else:
            asr_token_ids = asr_token_ids[:asr_len]

        # ocr
        ocr_tokens = self.tokenizer.tokenize(ocr_text)
        ocr_tokens = filter_tokens(ocr_tokens)
        ocr_token_ids = self.tokenizer.convert_tokens_to_ids(ocr_tokens)
        ocr_len = min(self.max_len - 4 - title_len - asr_len, len(ocr_tokens), self.max_ocr_len)
        if self.truncation == 'head_tail' and len(ocr_token_ids) > ocr_len:
            half1, half2 = ocr_len // 2, ocr_len - ocr_len // 2
            ocr_token_ids = ocr_token_ids[:half1] + ocr_token_ids[-half2:]
        else:
            ocr_token_ids = ocr_token_ids[:ocr_len]

        padding_len = self.max_len - 4 - title_len - asr_len - ocr_len
        token_ids = [101] + title_token_ids + [102] + asr_token_ids + [102] + ocr_token_ids + [102] + [
            0] * padding_len
        segment_ids = [0] * (title_len + 2) + [self.asr_type] * (asr_len + 1) + [self.ocr_type] * (
                ocr_len + 1) + [1] * padding_len
        attention_mask = [1] * (title_len + asr_len + ocr_len + 4) + [0] * padding_len

        input_token_ids = torch.LongTensor(token_ids)
        input_segment_ids = torch.LongTensor(segment_ids)
        input_attention_mask = torch.LongTensor(attention_mask)
        return input_token_ids, input_segment_ids, input_attention_mask


if __name__ == '__main__':
    anno_path = '../data/demo_data/annotations/semi_demo.json'
    zip_feat_path = '../data/demo_data/zip_feats/labeled.zip'
    dataset = Wechat_Dataset('../data/demo_data/zip_frames/demo', zip_feat_path=zip_feat_path,
                             anno_path=anno_path, phase='train', use_raw_image=True)
    print(len(dataset))
    # print(dataset[0])

    # dataloader = DataLoader(dataset, batch_size=8)
    #
    # for batch in dataloader:
    #     frame_input, frame_mask, title, asr, ocr, sample_id, label1, label2, tfidf = batch
    #     print(frame_input.size())
    #     # print(title)
    #     print(asr)
    #     print(ocr)
    #     # print(label1, label2)
    #     break

    num_frames = 6
    max_frame = 24
    index = list(range(0, num_frames, num_frames//max_frame if num_frames > max_frame else 1))[:max_frame]
    print(len(index), index)