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
from utils.util import evaluate


optiF1 = False

def inference(args):
    device = torch.device(args.device)
    time1 = time.time()
    # 2. load model
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model.half()
        model = torch.nn.parallel.DataParallel(model.cuda())
        
    model.eval()
    
    
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, args.train_zip_feats)
    val_index = [i for i in range(90000, 100000)]
    # val_index = [i for i in range(90000, 91000)]
    dataset = torch.utils.data.Subset(dataset, val_index)
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
    labels = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='valing...', total=len(dataloader)):
            frame_input, frame_mask = batch['frame_input'].to(device), batch['frame_mask'].to(device)
            title_input, title_mask = batch['title_input'].to(device), batch['title_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            pred_label_id = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
            # np.argmax(out1, axis=1)
            predictions.extend(np.argmax(pred_label_id.cpu().numpy(), axis=-1).reshape(-1,).tolist())
            labels.extend(batch['label'].numpy().reshape(-1,).tolist())
            count += 1
#             if count > 10:
#                 break
    
    results = evaluate(predictions, labels)
    print(results)
    # # 4. dump results
    # with open('./result_eval.csv', 'w') as f:
    #     for pred_label_id, ann in zip(predictions, dataset.anns):
    #         video_id = ann['id']
    #         category_id = lv2id_to_category_id(pred_label_id)
    #         f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    inference(args)
    
