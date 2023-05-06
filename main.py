import argparse
import sys, os
import numpy as np
import torch
from cfg import Cfg
from utils.data_prepare import prepare
from trainer import ArcfaceTrainer, RecognitionTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_train', action='store_true', help='prepare train data')
parser.add_argument('--data_test', action='store_true', help='prepare test data')
parser.add_argument('--arc_train', action='store_true', help='start arcface train')
parser.add_argument('--meta_train', action='store_true', help='start meta recognition train')

parser.add_argument('--weight_path', type=str, default=None)
opt = parser.parse_args()

def main(opt, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory=True
    if opt.data_train:
        prepare(use_train=True)
    elif opt.data_test:
        prepare(use_train=False)
    elif opt.arc_train:
        ArcfaceTrainer(config, device, num_workers=cfg.num_worker, pin_memory=pin_memory, weight_path=opt.weight_path)
    elif opt.meta_train:
        RecognitionTrainer(config, device, num_workers=cfg.num_worker, pin_memory=pin_memory, weight_path=opt.weight_path)
    #elif opt.meta_train:
        #RecognitionTrainer(config, device, num_workers=cfg.num_worker, pin_memory=pin_memory, weight_path=opt.weight_path)

if __name__ == '__main__':
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    try:
        main(opt, cfg)
    except KeyboardInterrupt:
        sys.exit(1)
        raise
