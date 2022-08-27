# encoding=utf-8
import argparse
import os, time, gc, sys, psutil

from tqdm import tqdm
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, './'))
from dataset.wx_dataset import WXDataset
from models.WXPreModel import WXModel
from utils.util import setup_seed
from torch.cuda.amp import GradScaler as GradScaler
os.environ["TOKENIZERS_PARALLELISM"] = "false"
base_dir = os.path.dirname(__file__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
from imp import reload

reload(logging)
# logger_path = '/home/tione/notebook/env/wbdc2022_semi/src/bph_src/data/loger_path/'
logger_path = os.path.join(base_dir, './data/logger_path')

os.makedirs(logger_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"{logger_path}/train_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
)

import numpy as np

sys.path.append(os.path.join(base_dir, '../'))
from pretrain.optim.create_optimizer import create_optimizer
from pretrain.config.model_cfg import *
from pretrain.config.pretrain_cfg import *
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast
gc.enable()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# DEVICE = "cpu"
input_grad=[]
output_grad=[]

def get_pred_and_loss(model, item, task=None):
    video_feature = item['frame_input'].to(DEVICE)
    input_ids = item['text_input'].to(DEVICE)
    attention_mask = item['mask'].to(DEVICE)
    video_mask = item['frame_mask'].to(DEVICE)

    pred, emb, loss, dic_loss = model(video_feature, video_mask, input_ids, attention_mask, task)
    return pred, emb, loss, dic_loss


def eval(model, data_loader, get_pred_and_loss, compute_loss=True, eval_max_num=99999):
    """Evaluates the |model| on |data_loader|"""
    model.eval()
    loss_l, emb_l, vid_l = [], [], []
    loss_mlm, loss_mfm, loss_itm, loss_itc = [], [], [], []
    loss_ml, loss_f, loss_i, loss_c = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        batch_num = 0.0
        for item in tqdm(data_loader, desc='valing...', total=len(data_loader)):
            pred, emb, loss, dic_loss = get_pred_and_loss(model, item)

            if loss is not None:
                loss_l.append(loss.to("cpu"))
            if 'mlm' in dic_loss:
                loss_mlm.append(dic_loss['mlm'].to("cpu").mean())
            if 'mfm' in dic_loss:
                loss_mfm.append(dic_loss['mfm'].to("cpu").mean())
            if 'itm' in dic_loss:
                loss_itm.append(dic_loss['itm'].to("cpu").mean())
            if 'itc' in dic_loss:
                loss_itc.append(dic_loss['itc'].to("cpu").mean())

            emb_l += emb.to("cpu").tolist()

            if (batch_num + 1) * emb.shape[0] >= eval_max_num:
                break
            batch_num += 1
        if 'mlm' in dic_loss:
            # print(loss_mlm)
            loss_ml = np.mean(loss_mlm)
        if 'mfm' in dic_loss:
            loss_f = np.mean(loss_mfm)
        if 'itm' in dic_loss:
            loss_i = np.mean(loss_itm)
        if 'itc' in dic_loss:
            loss_c = np.mean(loss_itc)
    return torch.mean(torch.stack(loss_l)), np.array(emb_l), loss_ml, loss_f, loss_i,loss_c

# 设置hook func
def hook_func(name, module):
    def hook_function(module, inputs, outputs):
        input_grad.append(inputs)
        output_grad.append(outputs)
        print('inputs',inputs[0].grad,'outputs',outputs[0].grad)
        # print('mean:',inputs.mean(),'min:',inputs.min(),inputs.max())
        # print('mean:',outputs.mean(),'min:',outputs.min(),outputs.max())

    return hook_function




def train(args, model, model_path, train_loader_l, val_loader, optimizer, get_pred_and_loss,
          scheduler=None, num_epochs=5
          ):

    if args.fp16:
        scaler = GradScaler()
        # cudnn.benchmark = True
        # 注册正反向hook
        # for name, module in model.named_modules():#named_parameters():
        #     module.register_forward_hook(hook_func('[forward]: '+name, module))
        #     module.register_backward_hook(hook_func('[backward]: '+name, module))

    else:
        logging.info('we do not use fp16')
        scaler = None
        
        # cudnn.benchmark = True
            
    best_val_loss, best_epoch, step = None, 0, 0
    start = time.time()
    for epoch in range(num_epochs):
        if epoch == args.early_stop:
            break
            
        model_path_epoch = f"signal_pytorch_model_{epoch}.bin"
        for train_loader, item, type_ in train_loader_l:
            batch_num = 0
            for item in tqdm(train_loader, desc=f"{type_}-{epoch}/{num_epochs} pretraining...",
                             total=len(train_loader)):
                model.train()
                
                optimizer.zero_grad()
                video_feature = item['frame_input'].to(DEVICE)
                input_ids = item['text_input'].to(DEVICE)
                attention_mask = item['mask'].to(DEVICE)
                video_mask = item['frame_mask'].to(DEVICE)
                
                if args.fp16:
                    with autocast():
                        pred, emb, loss, dic_loss = model(video_feature, video_mask, input_ids, attention_mask, task=None)
                        # print('--------------------loss--------------:',loss,dic_loss)
                        # logging.info('total loss:',loss,'dic loss:',dic_loss['mlm'],)
                        loss=loss.mean()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
#                 for name, parms in model.named_parameters():

#                     print('-->name:', name)
#                     print('-->para:', parms)
#                     print('-->grad_requirs:',parms.requires_grad)
#                     print('-->grad_value:',parms.grad)
                else:
                    pred, emb, loss, dic_loss = model(video_feature, video_mask, input_ids, attention_mask, task=None)
                    # print('--------------------loss--------------:',loss,dic_loss)
                    loss=loss.mean()
                    loss.backward()
                    optimizer.step()
                
                # pred, emb, loss, dic_loss = get_pred_and_loss(args, model, item)
                # loss=loss.mean()
                # loss.backward()

                # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

                # optimizer.step()
                
                if scheduler:
                    scheduler.step()

                if step == 20 or (step % 5000 == 0 and step > 0):
                    elapsed_seconds = time.time() - start  # Evaluate the model on val_loader.

                    val_loss, emb, loss_m, loss_f, loss_itm, loss_c = eval(
                        model, val_loader, get_pred_and_loss=get_pred_and_loss, eval_max_num=2000
                    )

                    improve_str = ''
                    if not best_val_loss or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # args.savedmodel_pat + '/' + model_path
                        # torch.save(model.state_dict(), args.savedmodel_path + '/' + model_path)
                        # torch.save(model.module.bert.state_dict(), args.savedmodel_path + '/best_pytorch_model.pth')
                        improve_str = f"|New best_val_loss={best_val_loss:6.4}"

                    logging.info(
                        f"Epoch={epoch + 1}/{num_epochs}|step={step:3}|val_loss={val_loss:6.4}|mlm_loss={loss_m}|"
                        f"mfm_loss={loss_f}|itm_loss={loss_itm}|itc_loss={loss_c}|"
                        f"time={elapsed_seconds:0.3}s" + improve_str)

                    start = time.time()
                step += 1
                batch_num += 1
        # torch.save(model.state_dict(), args.savedmodel_path + '/' + model_path_epoch)
        torch.save(model.module.bert.state_dict(), args.savedmodel_path + f'/pytorch_model.bin')
        # model.load_state_dict(torch.load(model_path_epoch))  # Load best model
        # torch.save(model.state_dict(), model_path)
        val_loss, emb, loss_m, loss_f, loss_itm, loss_c = eval(model, val_loader, get_pred_and_loss=get_pred_and_loss,
                                                       eval_max_num=99999)
        # label, spear = evaluate_emb_spearman(emb, vid_l, label_path=f"{DATA_PATH}/pairwise/label.tsv")
        logging.info(f"val_loss={val_loss:6.4}|mlm_loss={loss_m}|mfm_loss={loss_f}|itm_loss={loss_itm}|itc_loss={loss_c}")

    return best_val_loss


def main(args):
    # Show config
    logging.info(f"Start >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    logging.info(f"args: {args}")
    for fname in ['pretrain', 'model', 'data']:
        logging.info('=' * 66)
        with open(os.path.join(base_dir, f'./pretrain/config/{fname}_cfg.py')) as f:
            logging.info(f"Config - {fname}:" + '\n' + f.read().strip())

    list_val_loss = []
    logging.info(f"Model_type = {MODEL_TYPE}")

    for fold in range(NUM_FOLDS):
        logging.info('=' * 66)
        model_path = f"model_pretrain_{fold + 1}.pth"
        logging.info(f"Fold={fold + 1}/{NUM_FOLDS} seed={args.seed}")

        # Load dataset
        logging.info("Load data into memory")
        m0 = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30
        dataset = WXDataset(
            args, ann_path=args.train_annotation, zip_frame_dir=args.train_zip_frames, zip_feats=args.train_zip_feats
        )
        train_index, val_index = [i for i in range(90000)], [i for i in range(90000, 100000)]
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)
        

        if args.debug:
            train_index = [i for i in range(100)]
            train_dataset = torch.utils.data.Subset(train_dataset, train_index)
            val_dataset = torch.utils.data.Subset(val_dataset, train_index)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        task_ = ['mlm','itm','mfm']
        # task_ = ['mlm','itm','mfm', 'itc']
        # task_ = ['mfm']
        # task_=args.tasks
        total_steps = NUM_EPOCHS * (len(train_dataset)) // args.batch_size
        train_loader_l = [(train_loader, task_, 'train')]
        if args.use_unlabeled:
            unlabeled_dataset = WXDataset(
                args, ann_path=args.unlabeled_annotation, zip_frame_dir=args.unlabeled_zip_frames, zip_feats=args.unlabeled_zip_feats
            )

            if args.debug:
                train_index = [i for i in range(100)]
                unlabeled_dataset = torch.utils.data.Subset(unlabeled_dataset, train_index)

            unlabeled_loader = DataLoader(
                unlabeled_dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=args.num_workers
            )
            train_loader_l = None
            # train_loader_l = [(train_loader, task_, 'train'),
            #                   (unlabeled_loader, task_, 'unlabeled')]
            train_loader_l = [(unlabeled_loader, task_, 'unlabeled'),
                              (train_loader, task_, 'train')]
            total_steps = NUM_EPOCHS * (len(train_dataset) + len(unlabeled_dataset)) // args.batch_size

        delta_mem = psutil.Process(os.getpid()).memory_info()[0] / 2. ** 30 - m0
        logging.info(f"Dataset used memory = {delta_mem:.1f}GB")

        # total_steps = NUM_EPOCHS * (len(train_dataset) + len(unlabeled_dataset)) // args.batch_size
        warmup_steps = int(WARMUP_RATIO * total_steps)
        logging.info(f'Total train steps1={total_steps}, warmup steps1={warmup_steps}')

        # model
        model = WXModel(args, MODEL_CONFIG, task=task_)
        if DEVICE =='cuda':
            model = torch.nn.parallel.DataParallel(model.to(DEVICE))
        
        
        # print("加载权重")
        # model.load_state_dict(torch.load('./data/my_pretrain/pretrain_clip_end2end/signal_pytorch_model_0.bin'))  # Load best model
        # optimizer
        optimizer = create_optimizer(model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

        # schedueler
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps,
                                                    num_warmup_steps=warmup_steps)

        val_loss = train(args, model, model_path, train_loader_l, val_loader, optimizer,
                         get_pred_and_loss=get_pred_and_loss,
                         scheduler=scheduler, num_epochs=args.max_epochs)
        list_val_loss.append(val_loss)

        del train_dataset
        gc.collect()

        logging.info(f"Fold{fold} val_loss_list=" + str([round(kk.item(), 6) for kk in list_val_loss]))

    logging.info(f"Val Cv={np.mean(list_val_loss):6.4} +- {np.std(list_val_loss):6.4}")
    logging.info("Train finish")


def parse_args():
    data_path = '/opt/ml/input/'
    root_path = os.path.join(base_dir, '../../')
    parser = argparse.ArgumentParser(description="Weixin Challenge 2022 pretraining")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=5, help="num_workers")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--train_annotation', type=str, default=data_path + 'data/annotations/labeled.json')
    parser.add_argument('--train_zip_frames', type=str, default=data_path + 'data/zip_frames/labeled/')
    parser.add_argument('--train_zip_feats', type=str, default=data_path + 'data/zip_feats_clip/labeled.zip')
    parser.add_argument('--prefetch', default=4, type=int, help="use for training duration per worker")
    
    parser.add_argument('--test_annotation', type=str, default=data_path + 'data/annotations/semi_demo.json')
    parser.add_argument('--test_zip_frames', type=str, default=data_path + 'data/zip_frames/demo/')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    
    parser.add_argument('--unlabeled_annotation', type=str, default=data_path + 'data/annotations/unlabeled.json')
    parser.add_argument('--unlabeled_zip_frames', type=str, default=data_path + 'data/zip_frames/unlabeled/')
    parser.add_argument('--unlabeled_zip_feats', type=str, default=data_path + 'data/zip_feats_unlabeled/unlabeled.zip')

    # bert
    parser.add_argument('--bert_dir', type=str, default=data_path + 'env/wbdc2022_semi/data/pretrain_models/chinese-macbert-base')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--use_unlabeled', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--input_shape', default=[256,256], help='shape of imgs')
    parser.add_argument('--savedmodel_path', type=str, default=data_path + 'env/wbdc2022_semi/src/bph_src/data/my_pretrain')
    parser.add_argument('--loger_path', type=str, default=data_path + 'env/wbdc2022_semi/src/bph_src/data/loger_path')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=6, help='How many epochs')
    parser.add_argument('--end2end', action='store_true')
    parser.add_argument('--tasks', default=['mlm', 'mfm', 'itm'])
    parser.add_argument('--frame_emb_type', type=str, default='frame')
    parser.add_argument('--frame_encoder_config_path', type=str, default=root_path + 'data/pretrain_models/clip_vit_base_32/config.json')
    parser.add_argument('--frame_encoder_path', type=str, default=root_path + './data/pretrain_models/clip_vit_base_32/pytorch_model.bin')
    parser.add_argument('--early_stop', type=int, default=15)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.n_gpu = torch.cuda.device_count()
    setup_seed(args)
    os.makedirs(args.savedmodel_path, exist_ok=True)
    os.makedirs(args.loger_path, exist_ok=True)
    logging.info(args)
    main(args)

    # task_ = ['mlm', 'mfm', 'itm']
    # model = WXModel(args, MODEL_CONFIG, task=task_)
    # model.to(DEVICE)
    # model.load_state_dict(torch.load('./premodel_1/model_pretrain_epoch_6.pth'))  # Load best model
    # torch.save(model.bert.state_dict(), './premodel_1/bert_pretrain_1.pth')
    # conda create --prefix /home/tione/notebook/envs/wbdc2022_tiacc -y --clone tiacc_pytorch_py3 python=3.8
    # python -m torch.distributed.launch --nproc_per_node=2 pretrain1.py
