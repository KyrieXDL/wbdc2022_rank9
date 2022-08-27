python src/xdl_src/extract_feature.py\
    --zip_frame_dir '/opt/ml/input/data/zip_frames/unlabeled/'\
    --ann_path '/opt/ml/input/data/annotations/unlabeled_new.json'\
    --output_path '/opt/ml/input/data/zip_feats_unlabeled/unlabeled.zip'\
    --frame_encoder_path './opensource_models/pretrain_models/clip_vit_base_32/pytorch_model.bin'\
    --frame_encoder_config_path './opensource_models/pretrain_models/clip_vit_base_32/config.json'\
