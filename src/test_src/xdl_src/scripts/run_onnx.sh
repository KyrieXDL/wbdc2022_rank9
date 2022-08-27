python src/xdl_src/onnx.py \
    --flag 'base_clipvit_contras'\
    --zip_frame_path '/home/tione/notebook/data/zip_frames/labeled/' \
    --test_anno_path '/home/tione/notebook/data/annotations/val_labeled.json'\
    --device 'cuda' \
    --device_ids '0,1' \
    --phase 'test'\
    --model_save_path './src/xdl_src/saved_models/base_clipvit_contras'\
    --export_model_path './src/xdl_src/saved_models/base_clipvit_contras/model.onnx'\
    --output_dir './src/xdl_src/output/logs'\
    --batch_size 2\
    --accumu_grad_step 1\
    --epochs 10\
    --lr_pretrained 1e-5\
    --lr_random 5e-5\
    --warmup_steps 1000\
    --frame_encoder_arch 'clip_vit'\
    --frame_encoder_path './data/pretrain_models/clip_vit_base_32/pytorch_model.bin'\
    --frame_encoder_config_path './data/pretrain_models/clip_vit_base_32/config.json'\
    --visual_encoder_arch 'transformer_prenorm'\
    --visual_encoder_path ''\
    --visual_encoder_config_path './src/xdl_src/configs/visual_encoder_cls_config_small.json'\
    --text_encoder_arch 'bert'\
    --text_encoder_path './data/pretrain_models/chinese-macbert-base'\
    --multimodal_config_path './src/xdl_src/configs/cross_attention_config.json'\
    --fusion 'cross_attention' \
    --cross_type 'image_text' \
    --loss 'cross_entropy'\
    --schedule_type 'poly'\
    --max_title_len 90\
    --max_asr_len 90\
    --max_ocr_len 90\
    --max_frames 10\
    --use_asr\
    --use_ocr\
    --use_visual_encoder\
    --visual_embed_dim 768 \
    --text_embed_dim 768\
    --mm_embed_dim 768\
    --use_ema\
    --alpha 0.4\
    --momentum 0.999\
    --pooling ''\
    --spatial_dropout 0\
    --use_single_modal\
    --use_raw_image\
    --num_workers 8\
    --prefetch 8\
    --use_contrastive\
    --checkpoint './src/xdl_src/saved_models/base_clipvit_contras/checkpoint_m_2.pth'\
