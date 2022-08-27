echo "抽取特征"
mkdir /opt/ml/input/data/zip_feats_clip
mkdir /opt/ml/input/data/zip_feats_unlabeled

sh src/xdl_src/scripts/run_extract_feature.sh
sh src/xdl_src/scripts/run_extract_feature_unlabel.sh

echo "开始bph"
sh src/bph_src/scripts/train.sh


echo "开始xdl"
python src/xdl_src/split_data.py

sh src/xdl_src/scripts/run_pretrain_not_e2e.sh

sh src/xdl_src/scripts/run_pretrain_not_e2e_v2.sh

sh src/xdl_src/scripts/run_train_clipvit_contras_distill_not_e2e_v2.sh

sh src/xdl_src/scripts/run_train_clipvit_contras_not_e2e.sh

sh src/xdl_src/scripts/run_swa.sh

sh src/xdl_src/scripts/run_swa_v2.sh