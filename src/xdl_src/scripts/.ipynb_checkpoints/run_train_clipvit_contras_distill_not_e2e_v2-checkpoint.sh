python -m torch.distributed.run --nproc_per_node=2 src/xdl_src/train.py \
    --flag 'base_clipvit_contras_distill_not_e2e_v2'\
    --zip_frame_path '/opt/ml/input/data/zip_frames/labeled/' \
    --zip_feat_path '/opt/ml/input/data/zip_feats_clip/labeled.zip'\
    --anno_path '/opt/ml/input/data/annotations/labeled.json' \
    --device 'cuda' \
    --distributed\
    --dist_url 'tcp://127.0.0.1:28986'\
    --device_ids '0,1' \
    --phase 'train'\
    --model_save_path './src/xdl_src/saved_models/base_clipvit_contras_distill_not_e2e_v2'\
    --output_dir './src/xdl_src/output/logs'\
    --batch_size 16\
    --accumu_grad_step 1\
    --epochs 10\
    --lr_pretrained 1e-5\
    --lr_random 5e-5\
    --warmup_steps 1000\
    --frame_encoder_arch 'clip_vit'\
    --frame_encoder_path './opensource_models/pretrain_models/clip_vit_base_32/pytorch_model.bin'\
    --frame_encoder_config_path './opensource_models/pretrain_models/clip_vit_base_32/config.json'\
    --visual_encoder_arch 'transformer_prenorm'\
    --visual_encoder_path ''\
    --visual_encoder_config_path './src/xdl_src/configs/visual_encoder_cls_config_small.json'\
    --text_encoder_arch 'bert'\
    --text_encoder_path './opensource_models/pretrain_models/chinese-macbert-base'\
    --multimodal_config_path './src/xdl_src/configs/cross_attention_config.json'\
    --fusion 'cross_attention' \
    --cross_type 'image_text' \
    --loss 'cross_entropy'\
    --schedule_type 'poly'\
    --max_title_len 90\
    --max_asr_len 90\
    --max_ocr_len 90\
    --max_frames 32\
    --use_asr\
    --use_ocr\
    --use_visual_encoder \
    --visual_embed_dim 768 \
    --text_embed_dim 768\
    --mm_embed_dim 768\
    --use_ema\
    --alpha 0.4\
    --momentum 0.999\
    --pooling ''\
    --spatial_dropout 0\
    --use_single_modal\
    --num_workers 4\
    --prefetch 6\
    --use_contrastive\
    --val_ratio 0\
    --use_swa\
    --use_fp16\
    --use_momentum_ckpt\
    --use_distill\
    --checkpoint './src/xdl_src/saved_models/pretrain_not_e2e_v2/checkpoint_5.pth'\
    