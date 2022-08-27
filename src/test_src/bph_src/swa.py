import copy
import logging
import os
import time
import torch
from tqdm import tqdm

from config import parse_args
from dataset.data_helper import create_dataloaders
from models.model import MultiModal
from utils.util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate

from utils.functions_utils import swa
from callback.adversarial import FGM
import torch.nn.functional as F

def cal_loss(prediction, label):
    label = label.squeeze(dim=1)
    loss = F.cross_entropy(prediction, label)
    with torch.no_grad():
        pred_label_id = torch.argmax(prediction, dim=1)
        accuracy = (label == pred_label_id).float().sum() / label.shape[0]
    return loss, accuracy, pred_label_id, label

def validate(model, val_dataloader, device):
    model.eval()
    predictions = []
    labels = []
    losses = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='valing...', total=len(val_dataloader)):
            frame_input, frame_mask = batch['frame_input'].to(device), batch['frame_mask'].to(device)
            title_input, title_mask = batch['title_input'].to(device), batch['title_mask'].to(device)
            token_type_ids, label = batch['token_type_ids'].to(device), batch['label'].to(device)
            prediction = model(frame_input, frame_mask, title_input, title_mask, token_type_ids)
            loss, accuracy, pred_label_id, _ = cal_loss(prediction, label)
            # loss, _, pred_label_id, label, _ = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy().reshape(-1,).tolist())
            labels.extend(label.cpu().numpy().reshape(-1,).tolist())
            losses.append(loss.cpu().numpy())
            count +=1
            # if count > 10:
            #     break
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    print(results)
    return loss, results


def get_swa(args, swa_start=1, swa_end=100):
    # 1. load data
    device = torch.device(args.device)
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    swa_raw_model = copy.deepcopy(model)

    swa_raw_model = swa(swa_raw_model, args.savedmodel_path, swa_start=swa_start, swa_end=swa_end)

    if args.device == 'cuda':
        swa_raw_model = torch.nn.parallel.DataParallel(swa_raw_model.to(args.device))

    loss, results = validate(swa_raw_model, val_dataloader, device)
    results = {k: round(v, 4) for k, v in results.items()}
    logging.info(f"average swa: loss {loss:.3f}, {results}")

    # 5. save checkpoint
    mean_f1 = results['mean_f1']
    torch.save(
        {'epoch': args.max_epochs + 1, 'model_state_dict': swa_raw_model.module.state_dict(), 'mean_f1': mean_f1},
        f'{args.savedmodel_path}/model_epoch_swa_mean_f1_{mean_f1}_swa_{swa_start}_{swa_end}.bin')


def main():
    args = parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    # logging.info("Training/evaluation parameters: %s", args)
    get_swa(args, swa_start=args.swa_start, swa_end=args.swa_end)


if __name__ == '__main__':
    main()
