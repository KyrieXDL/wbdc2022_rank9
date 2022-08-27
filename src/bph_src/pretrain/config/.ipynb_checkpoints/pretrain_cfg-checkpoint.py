# Pretrain file num
PRETRAIN_FILE_NUM = 20
LOAD_DATA_TYPE = 'mem'#'fluid'
# Training params
NUM_FOLDS = 1
SEED = 42
NUM_EPOCHS = 5
WARMUP_RATIO = 0.06
REINIT_LAYER = 0
WEIGHT_DECAY = 0.01
LR = {'others':5e-4, 'bert':2e-5, 'newfc_videoreg':5e-4}
LR_LAYER_DECAY = 1.0
PRETRAIN_TASK = ['mlm', 'mfm']
