python -m torch.distributed.run --nproc_per_node=2 src/xdl_src/train.py \
    --flag 'base_crossatten_690'\
    --zip_frame_path '../demo_data/zip_frames/demo/' \
    --zip_feat_path '../demo_data/zip_feats/labeled.zip'\
    --train_anno_path '../demo_data/annotations/semi_demo.json' \
    --val_anno_path '../demo_data/annotations/semi_demo.json'\
    --test_anno_path '../demo_data/annotations/semi_demo.json'\
    --device 'cuda' \
    --distributed\
    --dist_url 'tcp://127.0.0.1:28764'\
    --device_ids '0,1' \
    --phase 'train'\
    --model_save_path './src/xdl_src/saved_models/base_crossatten_690'\
    --output_dir './src/xdl_src/output/logs'\
    --batch_size 2\
    --accumu_grad_step 1\
    --epochs 10\
    --lr_pretrained 1e-5\
    --lr_random 5e-5\
    --warmup_steps 1000\
    --visual_encoder_arch 'mvit'\
    --visual_encoder_path './data/pretrain_models/mvit_base_k400_32/MViTv2_B_32x3_k400_f304025456.pyth'\
    --visual_config_path './src/xdl_src/configs/mvit_base.yaml'\
    --text_encoder_arch 'bert'\
    --text_encoder_path './data/pretrain_models/chinese-macbert-base'\
    --multimodal_config_path './src/xdl_src/configs/cross_attention_config.json'\
    --fusion 'cross_attention' \
    --cross_type 'image_text' \
    --loss 'cross_entropy'\
    --schedule_type 'poly'\
    --max_title_len 100\
    --max_asr_len 100\
    --max_ocr_len 100\
    --use_asr\
    --use_ocr\
    --use_prompt\
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
    --use_raw_image\