# encoding=utf-8
from category_id_map import lv2id_to_category_id
import pickle

import json
import numpy as np


def get_logits(path):
    with open(path, 'r') as fr:
        lines = fr.readlines()

    logits = []
    ids = []
    for l in lines:
        item = json.loads(l)
        logits.append(item['logits'])
        ids.append(item['id'])

    return ids, np.array(logits)


def softmax(logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs


def merge2():
    ids, res1 = get_logits('data/results/base_crossatten_690_logits.jsonl')
    _, res2 = get_logits('data/results/base_crossatten_689_logits.jsonl')

    with open('data/results/result1.pkl', 'rb') as f:
        res3 = pickle.load(f)

    results = res3 * 0.5 + ((res1 + res2) / 2) * 0.5
    results = np.argmax(results, axis=1)
    with open('./data/result.csv', 'w') as f:
        for pred_label_id, video_id in zip(results, ids):
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    merge2()
