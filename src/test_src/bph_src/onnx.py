import time

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.util import setup_device, setup_seed, evaluate
import os
from config import parse_args
from models.model import MultiModal
from dataset.data_helper import MultiModalDataset
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler


def convert_onnx(model, inputs, export_model_path):
    # with torch.no_grad():
        # symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(model,  # model being run
                      args=inputs,  # model input (or a tuple for multiple inputs)
                      f=export_model_path,  # where to save the model (can be a file or file-like object)
                      opset_version=11,  # the ONNX version to export the model to
                      export_params=True,
                      training=False,
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['frame_input', 'frame_mask', 'title_input', 'title_mask', 'token_type_ids'],
                      output_names=['output'],  # the model's output names
                      verbose=False,
                      # dynamic_axes = {
                      #     'frame_input':{0, 'batch_size'},
                      #     'frame_mask':{0, 'batch_size'},
                      #     'title_input':{0, 'batch_size'},
                      #     'title_mask':{0, 'batch_size'},
                      #     'token_type_ids':{0, 'batch_size'},
                      #     'output':{0, 'batch_size'}
                      # }
                      )
    print("Model exported at ", export_model_path)

    
    
def main():
    args = parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    # setup_device(args)
    setup_seed(args)
    device = torch.device(args.device)
    # os.makedirs(args.savedmodel_path, exist_ok=True)

    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # if torch.cuda.is_available():
    #     model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()
    model.to(device)
    batch_size = 16
    inputs = (
        torch.randn((batch_size, 8, 3, 224, 224)).to(device),
        torch.randint(0, 2, (batch_size, 8)).to(device),
        torch.randint(0, 100, (batch_size, 256)).to(device),
        torch.randint(0, 2, (batch_size, 256)).to(device),
        torch.randint(0, 2, (batch_size, 256)).to(device),
    )
    export_model_path = './finetune-model.onnx'
    convert_onnx(model, inputs, export_model_path)


def infer():
    import onnxruntime
    export_model_path = './finetune-model.onnx'
    session = onnxruntime.InferenceSession(export_model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    args = parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_device(args)
    setup_seed(args)
    
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, args.train_zip_feats)
    val_index = [i for i in range(90000, 100000)]
    dataset = torch.utils.data.Subset(dataset, val_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            # batch_size=args.test_batch_size,
                            batch_size=16,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 3. inference
    predictions = []
    pres = []
    labels = []
    with torch.no_grad():
        step = 0
        for batch in dataloader:
            ort_inputs = {
                    'frame_input': batch['frame_input'].numpy(),
                    'frame_mask': batch['frame_mask'].numpy(),
                    'title_input': batch['title_input'].numpy(),
                    'title_mask': batch['title_mask'].numpy(),
                    'token_type_ids': batch['token_type_ids'].numpy(),
            }
         
            out1 = session.run(None, ort_inputs)[0]
            
            predictions.extend(out1)
            pres.extend(np.argmax(out1, axis=1).reshape(-1,).tolist())
            labels.extend(batch['label'].numpy().reshape(-1,).tolist())
            step +=1 
            # if step > 2:
            #     break
    
  
    results = evaluate(pres, labels)
    print(results)
    # predictions = np.array(predictions)
    # predictions = np.argmax(predictions, axis=1).tolist()
    # # 4. dump results
    # with open('./result_eval.csv', 'w') as f:
    #     for pred_label_id, ann in zip(predictions, dataset.anns):
    #         video_id = ann['id']
    #         category_id = lv2id_to_category_id(pred_label_id)
    #         f.write(f'{video_id},{category_id}\n')

    

if __name__ == '__main__':

    # 2、预训练
    # main()
    infer()






