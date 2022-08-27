echo '开始预训练roberta'

# env/wbdc2022_semi/data/pretrain_models/chinese-roberta-wwm-ext/config.json
python src/bph_src/pretrain1.py --savedmodel_path ./src/bph_src/data/my_pretrain/pretrain_clip_roberta --max_frames 32 --batch_size 32 --max_epochs 20  --use_unlabeled  --fp16 --bert_dir ./opensource_models/pretrain_models/chinese-roberta-wwm-ext --early_stop 15

echo '复制roberta配置文件'
cp ./opensource_models/pretrain_models/chinese-roberta-wwm-ext/config.json ./src/bph_src/data/my_pretrain/pretrain_clip_roberta/
cp ./opensource_models/pretrain_models/chinese-roberta-wwm-ext/vocab.txt ./src/bph_src/data/my_pretrain/pretrain_clip_roberta/
