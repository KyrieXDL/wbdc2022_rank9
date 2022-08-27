import torch
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
from tqdm import tqdm

from config import parse_args
from dataset.data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from models.model import MultiModal
from utils.functions_utils import OptimizedF1
import numpy as np
import pickle
import time
from functools import partial
import os

optiF1 = False

def inference(args):
    device = torch.device(args.device)
    time1 = time.time()
    # 2. load model
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
        
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

   

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            frame_input, frame_mask = batch['frame_input'].to(device), batch['frame_mask'].to(device)
            title_input, title_mask = batch['title_input'].to(device), batch['title_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            pred_label_id = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
            predictions.extend(pred_label_id.cpu().numpy())
    
    predictions = np.array(predictions)
    # with open('./result_flow_6849.pkl', 'wb') as f:
    #     pickle.dump(predictions, f)
    predictions = np.argmax(predictions, axis=1).tolist()
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    inference(args)
    
