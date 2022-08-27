echo "开始微调macbert large"


python -m torch.distributed.run --nproc_per_node=2 ./src/bph_src/main_large.py --savedmodel_path ./src/bph_src/data/checkpoint/clip_macbert_large --max_frames 32 --val_batch_size 16 --batch_size 16 --backbone clip_vit --max_epochs 10 --device_ids '0,1' --device 'cuda' --dist_url 'tcp://127.0.0.1:2116' --distributed --use_ema True --learning_rate 2e-5 --fp16 --num_workers 4 --bert_dir './src/bph_src/data/my_pretrain/pretrain_clip_large' --fusion_layer 18 --early_stop 4



echo "开始swa"
python ./src/bph_src/swa_large.py --savedmodel_path ./src/bph_src/data/checkpoint/clip_macbert_large --max_frames 32 --val_batch_size 64 --batch_size 64 --backbone clip_vit --max_epochs 4 --device_ids '0,1' --device 'cuda' --swa_start 1 --swa_end 4 --prefetch 8 --num_workers 4 --bert_dir './src/bph_src/data/my_pretrain/pretrain_clip_large' --fusion_layer 18







