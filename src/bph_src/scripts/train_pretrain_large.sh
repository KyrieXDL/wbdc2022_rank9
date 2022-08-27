echo '开始预训练macbert large'

# 32, 32
python ./src/bph_src/pretrain_large.py --savedmodel_path ./src/bph_src/data/my_pretrain/pretrain_clip_large --max_frames 32 --batch_size 32 --max_epochs 20  --use_unlabeled  --fp16 --bert_dir ./opensource_models/pretrain_models/chinese-macbert-large --fusion_layer 18 --early_stop 10

cp ./opensource_models/pretrain_models/chinese-macbert-large/config.json ./src/bph_src/data/my_pretrain/pretrain_clip_large/
cp ./opensource_models/pretrain_models/chinese-macbert-large/vocab.txt ./src/bph_src/data/my_pretrain/pretrain_clip_large/


