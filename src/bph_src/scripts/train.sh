# 预训练
# train_pretrain_large
sh src/bph_src/scripts/train_pretrain_large.sh
# 问题：之前一个模型py文件，删除了，代码里没有删除
sh src/bph_src/scripts/train_pretrain_roberta.sh

# # 微调
sh src/bph_src/scripts/train_macbert_large.sh
sh src/bph_src/scripts/train_roberta.sh