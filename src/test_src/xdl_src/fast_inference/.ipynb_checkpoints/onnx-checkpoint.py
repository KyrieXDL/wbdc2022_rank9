import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import sys
sys.path.append('..')
from models.multimodal_classifiser import MultiModal_Classifier
from mydataset.category_id_map import lv2id_to_category_id
from mydataset.wechat_datatset import Wechat_Dataset


def convert_onnx(model, inputs, export_model_path):
    with torch.no_grad():
        # symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,  # model being run
                          inputs,  # model input (or a tuple for multiple inputs)
                          f=export_model_path,  # where to save the model (can be a file or file-like object)
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['frame_input', 'frame_mask', 'text_input_ids', 'text_segment_ids', 'text_attention_mask'],
                          output_names=['output'],  # the model's output names
                          verbose=False,
                          dynamic_axes={'frame_input': {0: 'batch_size'},  # variable length axes
                                        'frame_mask': {0: 'batch_size'},
                                        'text_input_ids': {0: 'batch_size'},
                                        'text_segment_ids': {0: 'batch_size'},
                                        'text_attention_mask': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}}
                          )
    print("Model exported at ", export_model_path)


def main(args):
    # os.makedirs(args.savedmodel_path, exist_ok=True)
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
              'use_single_modal': args.use_single_modal, 'use_raw_image': args.use_raw_image, 'use_aug': args.use_aug}
    if args.rank == 0:
        print(config)

    model = MultiModal_Classifier(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint: ', msg)
    # model = model.to(device)
    model.eval()
    
    print('model device: ', model.parameters().device)
    inputs = (torch.randn((1, 10, 3, 224, 224)),
              torch.randint(0, 2, (1, 10)),
              torch.randint(0, 100, (1, 270)),
              torch.randint(0, 100, (1, 270)),
              torch.randint(0, 100, (1, 270)))
    convert_onnx(model, inputs, args.export_model_path)


def infer(args):
    import onnxruntime
    # export_model_path = './data/checkpoint/model_0712/baseline_v1/finetune-model.onnx'
    session = onnxruntime.InferenceSession(args.export_model_path,
                                           providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                      'CPUExecutionProvider'])

    test_dataset = Wechat_Dataset(zip_frame_path=args.zip_frame_path, zip_feat_path=args.zip_feat_path,
                                  anno_path=args.test_anno_path, use_raw_image=args.use_raw_image,
                                  max_frames=args.max_frames,
                                  args=args)

    # train_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [75000, 25000],
    #                                                            generator=torch.Generator().manual_seed(args.seed))
    print(len(test_dataset))

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                                 shuffle=False, num_workers=args.num_workers,
                                 prefetch_factor=args.prefetch)

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            frame_feat, frame_feat_mask = batch[0], batch[1]
            text_input_ids, text_segment_ids, text_attention_mask = batch[2], batch[3], batch[4]
            sample_id = batch[5]
            ort_inputs = {
                'frame_input': frame_feat.numpy(),
                'frame_mask': frame_feat_mask.numpy(),
                'text_input_ids': text_input_ids.numpy(),
                'text_segment_ids': text_segment_ids.numpy(),
                'text_attention_mask': text_attention_mask.numpy(),
            }

            out1 = session.run(None, ort_inputs).item()
            predictions.extend(out1.cpu().numpy())

    predictions = np.array(predictions)
    predictions = np.argmax(predictions, axis=1).tolist()
    # 4. dump results
    # with open('./result_eval.csv', 'w') as f:
    #     for pred_label_id, ann in zip(predictions, dataset.anns):
    #         video_id = ann['id']
    #         category_id = lv2id_to_category_id(pred_label_id)
    #         f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output/logs')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/base')
    parser.add_argument('--export_model_path', type=str, default='./saved_models/finetune-model.onnx')
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

    main(args)
    # infer()
