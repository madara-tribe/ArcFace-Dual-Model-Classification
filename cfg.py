import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.pin_memory=True
Cfg.num_worker = 4
Cfg.width = 260
Cfg.height = 260
Cfg.train_batch = 4
Cfg.val_batch = 1
Cfg.epochs = 200
Cfg.num_classes = 122
Cfg.num_color = 11
Cfg.num_shape=2
Cfg.eval_step = 20
Cfg.val_interval = 2000
Cfg.gpu_id = '3'
Cfg.input_size=Cfg.height
Cfg.loss = 'focal_loss'
Cfg.loss="ArcFace"
if (Cfg.loss=="ArcFace"):
    Cfg.s = 30
    Cfg.m = 0.35
Cfg.embedding_size = 512

#Cfg.TRAIN_OPTIMIZER="sgd"
Cfg.TRAIN_OPTIMIZER = 'adam'
if (Cfg.TRAIN_OPTIMIZER=="adam"):
    Cfg.weight_decay = 1e-4
    Cfg.momentum = 0.9
    Cfg.lr = 0.01
elif (Cfg.TRAIN_OPTIMIZER=="sgd"):
    Cfg.weight_decay = 1e-4
    Cfg.momentum = 0.9
    Cfg.lr = 0.01
Cfg.T_0=50      
Cfg.T_mult=2
Cfg.eta_min=0.001

## dataset
Cfg.X_train = "dataset/npy/X_train.npy"
Cfg.y_train = "dataset/npy/cls_train.npy"
Cfg.color_train="dataset/npy/color_train.npy"
Cfg.shape_train="dataset/npy/shape_train.npy"
Cfg.embed_X="dataset/npy/X_embbed.npy"
Cfg.embed_y = "dataset/npy/y_embbed.npy"


Cfg.X_test = "dataset/npy/X_test.npy"
Cfg.y_test = "dataset/npy/cls_test.npy"
Cfg.color_test="dataset/npy/color_test.npy"
Cfg.shape_test="dataset/npy/shape_test.npy"

Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')

