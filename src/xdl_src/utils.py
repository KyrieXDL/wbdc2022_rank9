# from transformers.optimization import AdamW
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup
import logging
import os
import torch
import torch.distributed as dist
import numpy as np
import random
import copy


def create_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    random.seed(seed)


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


def create_optimizer(model, lr_pretrained=1e-5, lr_random=5e-5, no_decay_names=[], pretrained_names=[],
                     warmup_steps=1000, max_steps=100000, schedule_type='warmup'):
    weight_decay = 0.01

    # if len(random_names) == 0:
    #     random_names = ["cross_attention", "mlm_head", "itm_head", "vision_proj", "text_proj"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_names)
                   and any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr_pretrained,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_names)
                   and any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": 0.0,
            "lr": lr_pretrained,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_names)
                   and not any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": weight_decay,
            "lr": lr_random,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_names)
                   and not any(bb in n for bb in pretrained_names)
            ],
            "weight_decay": 0.0,
            "lr": lr_random,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=lr_random, eps=1e-8, betas=(0.9, 0.98)
    )

    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)
    print('warmup steps: {} | max steps: {}'.format(warmup_steps, max_steps))

    if schedule_type == 'warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=max_steps)
    elif schedule_type == 'poly':
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=0,
            power=1,
        )
    else:
        scheduler = None

    return optimizer, scheduler


@torch.no_grad()
def momentum_update(model_pair, momentum):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data = param_m.data * momentum + param.data * (1. - momentum)


@torch.no_grad()
def copy_params(model_pair):
    for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
        param_m.data.copy_(param.data)  # initialize
        param_m.requires_grad = False  # not update by gradient


def get_title_asr_ocr_len(title_token_ids, asr_token_ids, ocr_token_ids, max_title_len, max_asr_len, max_ocr_len):
    if len(title_token_ids) <= max_title_len and len(asr_token_ids) <= max_asr_len and len(
            ocr_token_ids) <= max_ocr_len:
        title_len = len(title_token_ids)
        asr_len = len(asr_token_ids)
        ocr_len = len(ocr_token_ids)
    elif len(title_token_ids) > max_title_len and len(asr_token_ids) <= max_asr_len and len(
            ocr_token_ids) <= max_ocr_len:
        title_len = min(max_title_len + max_asr_len - len(asr_token_ids) +
                        max_ocr_len - len(ocr_token_ids), len(title_token_ids))
        asr_len = len(asr_token_ids)
        ocr_len = len(ocr_token_ids)
    elif len(title_token_ids) <= max_title_len and len(asr_token_ids) > max_asr_len and len(
            ocr_token_ids) <= max_ocr_len:
        title_len = len(title_token_ids)
        asr_len = min(max_asr_len + max_title_len - len(title_token_ids) +
                      max_ocr_len - len(ocr_token_ids), len(asr_token_ids))
        ocr_len = len(ocr_token_ids)
    elif len(title_token_ids) <= max_title_len and len(asr_token_ids) <= max_asr_len and len(
            ocr_token_ids) > max_ocr_len:
        title_len = len(title_token_ids)
        asr_len = len(asr_token_ids)
        ocr_len = min(max_ocr_len + max_title_len - len(title_token_ids) +
                      max_asr_len - len(asr_token_ids), len(ocr_token_ids))
    elif len(title_token_ids) > max_title_len and len(asr_token_ids) > max_asr_len and len(
            ocr_token_ids) <= max_ocr_len:
        diff_title_len = abs(max_title_len - len(title_token_ids))
        diff_asr_len = abs(max_asr_len - len(asr_token_ids))
        ratio = diff_title_len / (diff_asr_len + diff_title_len)

        title_len = min(max_title_len + int((max_ocr_len - len(ocr_token_ids)) * ratio), len(title_token_ids))
        asr_len = min(max_asr_len + int((max_ocr_len - len(ocr_token_ids)) * (1 - ratio)), len(asr_token_ids))
        ocr_len = len(ocr_token_ids)
    elif len(title_token_ids) > max_title_len and len(asr_token_ids) <= max_asr_len and len(
            ocr_token_ids) > max_ocr_len:
        diff_title_len = abs(len(title_token_ids) - max_title_len)
        diff_ocr_len = abs(len(ocr_token_ids) - max_ocr_len)
        ratio = diff_title_len / (diff_ocr_len + diff_title_len)

        title_len = min(max_title_len + int((max_asr_len - len(asr_token_ids)) * ratio), len(title_token_ids))
        asr_len = len(asr_token_ids)
        ocr_len = min(max_ocr_len + int((max_asr_len - len(asr_token_ids)) * (1 - ratio)),
                      len(ocr_token_ids))
    elif len(title_token_ids) <= max_title_len and len(asr_token_ids) > max_asr_len and len(
            ocr_token_ids) > max_ocr_len:
        diff_asr_len = abs(len(asr_token_ids) - max_asr_len)
        diff_ocr_len = abs(len(ocr_token_ids) - max_ocr_len)
        ratio = diff_asr_len / (diff_ocr_len + diff_asr_len)

        title_len = len(title_token_ids)
        asr_len = min(max_asr_len + int((max_title_len - len(title_token_ids)) * ratio), len(asr_token_ids))
        ocr_len = min(max_ocr_len + int((max_title_len - len(title_token_ids)) * (1 - ratio)),
                      len(ocr_token_ids))
    else:
        title_len = max_title_len
        asr_len = max_asr_len
        ocr_len = max_ocr_len

    return title_len, asr_len, ocr_len


def swa(model, model_dir, swa_start=1, swa_end=100):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = os.listdir(model_dir)
    model_path_list = [os.path.join(model_dir, f) for f in model_path_list if '_m_' in f and f.endswith('.pth')]
    model_path_list = sorted(model_path_list)
    print(model_path_list)

    assert 0 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:swa_end]:
            print(_ckpt)
            # logger.info(f'Load model from {_ckpt}')
            checkpoint = torch.load(_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            # model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    # swa_model_dir = model_dir
    # if not os.path.exists(swa_model_dir):
    #     os.mkdir(swa_model_dir)

    # logger.info(f'Save swa model in: {swa_model_dir}')
    # swa_model_path = os.path.join(swa_model_dir, 'model.pt')
    # torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model
