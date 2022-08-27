import copy
import os

import torch
import sys
from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '../'))


def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if '.bin' in _file and 'swa' not in _file:
                model_lists.append(os.path.join(root, _file).replace("\\", '/'))
    model_lists = sorted(model_lists,
                         key=lambda x: (x.split('/')[-1],
                                        int(x.split('/')[-1].split('.')[0].split('_')[-1])))
    return model_lists


def swa(model, model_dir, swa_start=1, swa_end=100):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)
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
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = model_dir
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    # logger.info(f'Save swa model in: {swa_model_dir}')
    #swa_model_path = os.path.join(swa_model_dir, 'model.pt')
    # torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model
    

class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef * X_p
        ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(200)]  # 权重都初始化为1
        # r _ in range(len(set(y)))]  # 权重都初始化为1
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')

    def coefficients(self):
        return self.coef_['x']


if __name__ == '__main__':
    get_model_path_list(r'F:\code\python\challenge-main\save\bert_v5')

    pass