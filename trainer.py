import logging
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchsummary import summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2

from cfg import Cfg
from utils.dataloader import DataLoader, MetaDataLoader
from utils.optimizers import create_optimizers
from utils.callback import CallBackModelCheckpoint
from models.BaseModel import ArcFaceModel, RecognitionModel
from models import metrics
from utils.focal_loss import FocalLoss

class ArcfaceTrainer:
    def __init__(self, config, device, num_workers, pin_memory, weight_path):
        global logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        self.global_val_loss = 0
        self.setup_resouces(config, device, num_workers, pin_memory)
        self.train(config, device, weight_path=weight_path)

    def setup_resouces(self, config, device, num_workers, pin_memory):
        self.writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}')
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        #if config.loss == 'focal_loss':
         #   self.criterion = FocalLoss(gamma=2)
        #else:
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.create_data_loader(config, use_imagefolder=False)
        self.callback_checkpoint = CallBackModelCheckpoint(config)
        
    def create_data_loader(self, config, use_imagefolder):
        """ Dataset And Augmentation
        if 0 ~ 1 normalize, just use:
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
        """
        train_transform = A.Compose([
            A.RandomBrightnessContrast(),
            #Random Erasing
            A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2()
            ])

        val_transform = A.Compose([
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2()
            ])
        if use_imagefolder:
            self.train_dst = ImageFolder("datasets/dataset", transform = train_transform)
            self.val_dst = ImageFolder("datasets/dataset", transform=val_transform)
        else:
            self.train_dst = DataLoader(config.X_train, config.y_train, transform=train_transform)
            self.val_dst = DataLoader(config.X_test, config.y_test, transform=val_transform)


        self.train_loader = data.DataLoader(
                self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = data.DataLoader(
                    self.val_dst, batch_size=config.val_batch, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))

    def validate(self, backbone, header, global_step, epoch, device):
        interval_valloss = 0
        backbone.eval()
        header.eval()
        print("validating .....")
        with torch.no_grad():
            for i, (x_val, y_val) in tqdm(enumerate(self.val_loader)):
                x_val = x_val.to(device=device)
                label = y_val.to(device=device).long()
                
                features = backbone(x_val)
                output = header(features, label)
                val_loss = self.criterion(output, label)
                interval_valloss += val_loss.item() / len(self.val_loader)

            self.writer.add_scalar('valid/interval_loss', interval_valloss, global_step)
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss))
            self.global_val_loss = interval_valloss
    
            self.callback_checkpoint(global_step, np.round(self.global_val_loss, decimals=4), backbone)
            logging.info(f'Checkpoint {epoch} saved!')
            backbone.train()
            header.train()
            return backbone, header
            
    def call_arcface_loss(self, cfg, device):
        if cfg.loss == "ArcFace":
            header = metrics.ArcMarginProduct(cfg.embedding_size, cfg.num_classes, device, s=cfg.s, m=cfg.m, easy_margin=False)
        else:
            header = nn.Linear(cfg.embedding_size, cfg.num_classes)
        return header

    def train(self, config, device, weight_path=None):
        # model
        backbone = ArcFaceModel(embedding_size=config.embedding_size).to(device)
        header = self.call_arcface_loss(config, device).to(device)
        #print(backbone)
         
        if weight_path is not None:
            backbone.load_state_dict(torch.load(weight_path, map_location=device))
        #if torch.cuda.device_count() > 1:
           # backbone = nn.DataParallel(backbone)
        summary(backbone, (3, config.input_size, config.input_size))
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Learning rate:   {config.lr}
            Training size:   {len(self.train_dst)}
            Validation size: {len(self.val_dst)}
        ''')
        
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        opt_backbone, scheduler_backbone = create_optimizers(backbone, config)
        opt_header, scheduler_header = create_optimizers(header, config)
        
        # 5. Begin training
        global_step = 0
        iters = len(self.train_loader)
        for epoch in range(1, config.epochs+1):
            interval_loss = 0
            backbone.train()
            header.train()
            with tqdm(total=int(len(self.train_dst)/config.train_batch), desc=f'Epoch {epoch}/{config.epochs}') as pbar:
                for i, (x_img, label) in enumerate(self.train_loader):
                    x_img = x_img.to(device=device)
                    label = label.to(device=device).long()
                    features = backbone(x_img)
                    #print(x_img.shape, x_img.min(), x_img.max(), y.shape, pred.shape)
                    output = header(features, label)
                    loss = self.criterion(output, label)
                    interval_loss += loss.item()
                    loss.backward()
                    
                    opt_backbone.step()
                    opt_header.step()
                    
                    scheduler_backbone.step(epoch + i / iters)
                    scheduler_header.step(epoch + i / iters)
                    
                    opt_backbone.zero_grad()
                    opt_header.zero_grad()

                    pbar.update()
                    global_step += 1
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
        
                    # Evaluation round
                    if global_step % config.eval_step == 0:
                        self.writer.add_scalar('train/train_loss', interval_loss/config.eval_step, global_step)
                        print("Epoch %d, Itrs %d, Loss=%f" % (epoch, global_step, interval_loss/config.eval_step))
                        interval_loss = 0

                    if global_step % config.val_interval == 0:
                        backbone, header = self.validate(backbone, header, global_step, epoch, device)
                        
 

class RecognitionTrainer:
    def __init__(self, config, device, num_workers, pin_memory, weight_path):
        global logging
        self.global_val_loss = 0
        self.setup_resouces(config, device, num_workers, pin_memory)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.info(f'Using device {device}')
        self.train(config, device, weight_path=weight_path)

    def setup_resouces(self, config, device, num_workers, pin_memory):
        self.writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.lr}_BS_Size_{config.width}')
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.create_data_loader(config, use_imagefolder=None)
        self.callback_checkpoint = CallBackModelCheckpoint(config)

    def create_data_loader(self, config, use_imagefolder=None):
        """ Dataset And Augmentation
        if 0 ~ 1 normalize, just use:
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        ToTensorV2()
        """
        train_transform = A.Compose([
            A.RandomBrightnessContrast(),
            #Random Erasing
            A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])

        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
        if use_imagefolder:
            self.train_dst = ImageFolder("datasets/dataset", transform = train_transform)
            self.val_dst = ImageFolder("datasets/dataset", transform=val_transform)
        else:
            self.train_dst = MetaDataLoader(config.X_train, config.color_train, config.shape_train, transform=train_transform)
            self.val_dst = MetaDataLoader(config.X_test, config.color_test, config.shape_test, transform=val_transform)
        
        self.train_loader = data.DataLoader(
                self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = data.DataLoader(
                    self.val_dst, batch_size=config.val_batch, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))

    def validate(self, model, global_step, epoch, device):
        color_vloss, shape_vloss = 0, 0
        model.eval()
        print("validating .....")
        with torch.no_grad():
            for i, (x_img, color, shape) in tqdm(enumerate(self.val_loader)):
                x_img = x_img.to(device=device)
                color = color.to(device=device)
                shape = shape.to(device=device)
                color_pred, shape_pred = model(x_img)
                val_closs = self.criterion(color_pred, color)
                val_sloss = self.criterion(shape_pred, shape)
                
                color_vloss += val_closs.item() / len(self.val_loader)
                shape_vloss += val_sloss.item() / len(self.val_loader)
            val_loss = color_vloss + shape_vloss
            
            self.writer.add_scalar('valid/color_loss', color_vloss, global_step)
            self.writer.add_scalar('valid/shape_loss', shape_vloss, global_step)
            print("Epoch %d, Itrs %d, valid_color_Loss=%f, valid_shape_Loss=%f" % (epoch, global_step, color_vloss, shape_vloss))
            self.global_val_loss = val_loss
            
            self.callback_checkpoint(global_step, np.round(self.global_val_loss, decimals=4), model)
            logging.info(f'Checkpoint {epoch} saved!')
            model.train()
            return model

    def train(self, config, device, weight_path=None):
        # model
        model = RecognitionModel(color_size=config.num_color, shape_size=config.num_shape).to(device)
        print(model)
        #"""
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        #if torch.cuda.device_count() > 1:
           # backbone = nn.DataParallel(backbone)
        summary(model, (3, config.input_size, config.input_size))
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Learning rate:   {config.lr}
            Training size:   {len(self.train_dst)}
            Validation size: {len(self.val_dst)}
        ''')
        
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        opt_backbone, scheduler_backbone = create_optimizers(model, config)

        # 5. Begin training
        global_step = 0
        iters = len(self.train_loader)
        for epoch in range(1, config.epochs+1):
            color_losses, shape_losses = 0, 0
            model.train()
            with tqdm(total=int(len(self.train_dst)/config.train_batch), desc=f'Epoch {epoch}/{config.epochs}') as pbar:
                for i, (x_img, color, shape) in enumerate(self.train_loader):
                    global_step += 1
                    x_img = x_img.to(device=device)
                    color = color.to(device=device)
                    shape = shape.to(device=device)
                    color_pred, shape_pred = model(x_img)
#                    print(x_img.shape, x_img.min(), x_img.max(), label.shape, pred.shape)
                    color_loss = self.criterion(color_pred, color)
                    shape_loss = self.criterion(shape_pred, shape)
                    loss = color_loss + shape_loss
                    color_losses += color_loss.item()
                    shape_losses += shape_loss.item()
                    
                    loss.backward()
                    opt_backbone.step()
                    scheduler_backbone.step(epoch + i / iters)
                    opt_backbone.zero_grad()
                    pbar.update()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    # Evaluation round
                    if global_step % config.eval_step == 0:
                        self.writer.add_scalar('train/colorLoss', color_losses/config.eval_step, global_step)
                        self.writer.add_scalar('train/shapeLoss', shape_losses/config.eval_step, global_step)
                        print("Epoch %d, Itrs %d, colorLoss=%f, shapeLoss=%f" % (epoch, global_step, color_losses/config.eval_step, shape_losses/config.eval_step))
                        color_losses, shape_losses = 0, 0

                    if global_step % config.val_interval == 0:
                        model = self.validate(model, global_step, epoch, device)
      
