import logging
import os.path
import math
import random
import time

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from category_id_map import lv2id_to_lv1id, lv1id_to_category_id_2

import torch.distributed as dist


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-12)


@torch.no_grad()
def copy_params(model_pair):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data.copy_(param.data)  # initialize
        param_m.requires_grad = False  # not update by gradient


@torch.no_grad()
def momentum_update(model_pair, momentum):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data = param_m.data * momentum + param.data * (1. - momentum)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu)
    
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    torch.cuda.set_device(args.gpu)
    
    
def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 


def setup_logging(args):
    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_file = os.path.join(args.savedmodel_path, f'{args.model_type}-{time_}.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger


def build_optimizer(args, model, t_total, T_mult=1, rewarm_epoch_num=1):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    no_decay_param_tp = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n and "visual_backbone." not in n]
    no_decay_vis_param_tp = [(n, p) for n, p in no_decay_param_tp if "visual_backbone." in n]
    
    #
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n and "visual_backbone." not in n]
    decay_vis_param_tp = [(n, p) for n, p in decay_param_tp if "visual_backbone." in n]
    
    

    optimizer_grouped_parameters = [  # 分层设置学习率
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.learning_rate},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.learning_rate},
        
        {'params': [p for n, p in no_decay_vis_param_tp], 'weight_decay': 0.01, 'lr': 1e-5},
        {'params': [p for n, p in decay_vis_param_tp], 'weight_decay': 0.0, 'lr': 1e-5},
        
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01, 'lr': 5e-4},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0, 'lr': 5e-4}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=0.0005)
    
    
    total_steps = t_total
    WARMUP_RATIO = 0.1
    warmup_steps = int(WARMUP_RATIO * total_steps)
    
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    T_mult = 1
    scheduler = CosineAnnealingWarmRestarts(optimizer, t_total // args.max_epochs * rewarm_epoch_num, T_mult, eta_min=5e-6, last_epoch=-1)
    # scheduler = MyWarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'mean_f1': mean_f1}

    return eval_results


def evaluate_2(predictions1, predictions2, labels1, label2):
    # prediction and labels are all level-2 class ids

    lv1_f1_micro = f1_score(labels1, predictions1, average='micro')
    lv1_f1_macro = f1_score(labels1, predictions1, average='macro')

    lv2_f1_micro = f1_score(label2, predictions2, average='micro')
    lv2_f1_macro = f1_score(label2, predictions2, average='macro')

    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(labels1, predictions1),
                    'lv2_acc': accuracy_score(label2, predictions2),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results
    
    
class MyWarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(MyWarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
