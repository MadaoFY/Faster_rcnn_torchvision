import os
import math
import yaml
import json
import torch
import cv2 as cv
import numpy as np


__all__ = [
    'yaml_load',
    'json_load',
    'same_seeds',
    'SineAnnealingLR',
    'mk_results_dirs',
]


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def json_load(file='data.json'):
    with open(file, "r") as f:
        return json.load(f)


def same_seeds(seed=42):
    """
        固定随机种子
    """
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    # torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    # torch.backends.cudnn.deterministic = True  # 固定网络结构


def SineAnnealingLR(opt, t_max):
    """
        sine学习率变化
    """

    lr_lambda = lambda x: (1 + math.cos(math.pi * x / t_max + math.pi)) * 0.5
    lr_sine = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    return lr_sine


def mk_results_dirs(root, folder='exp'):
    f = folder
    while os.path.exists(f'{root}/{f}'):
        print('File folder is already exists：', f'{root}/{f}')
        l = [1]
        for i in os.listdir(f'{root}'):
            try:
                l.append(int(i[len(f):]))
            except ValueError:
                continue
        n = max(l) + 1
        f = f'{f}{n}'
    print('New file folder is made：', f'{root}/{f}')
    os.makedirs(f'{root}/{f}')
    os.makedirs(f'{root}/{f}/labels')

    return f'{root}/{f}'