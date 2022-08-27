echo '开始训练 roberta'

python -m torch.distributed.run --nproc_per_node=2 src/bph_src/main.py --savedmodel_path ./src/bph_src/data/checkpoint/clip_roberta_base --max_frames 32 --val_batch_size 16 --batch_size 16 --backbone clip_vit --max_epochs 5 --device_ids '0,1' --device 'cuda' --dist_url 'tcp://127.0.0.1:1114' --distributed --use_ema True --learning_rate 5e-5 --fp16 --num_workers 6 --use_fgm True --bert_dir './src/bph_src/data/my_pretrain/pretrain_clip_roberta' --contras


echo "开始roberta swa"
python src/bph_src/swa.py --savedmodel_path ./src/bph_src/data/checkpoint/clip_roberta_base --max_frames 32 --val_batch_size 64 --batch_size 64 --backbone clip_vit --max_epochs 4 --device_ids '0,1' --device 'cuda' --swa_start 1 --swa_end 5 --prefetch 8 --num_workers 4 --bert_dir './src/bph_src/data/my_pretrain/pretrain_clip_roberta'
