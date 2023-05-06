import argparse
import sys, os
import numpy as np
import json
import time
import torch
from torch.utils import data
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataloader import DataLoader, MetaDataLoader
from models.BaseModel import ArcFaceModel, RecognitionModel
from cfg import Cfg    
from utils.metric import cosin_metric, cosine_similarity
from statistics import mode

root="dataset/npy"
X_embed ="X_embbed.npy"
y_embed ="y_embbed.npy"
# params
dd = json.load(open("dataset/cs_label.json"))


class BasicPredictor:
    def __init__(self, cfg, device, weight_path):
        self.cfg = cfg
        self.device = device
        self.weight_path = weight_path
        
    def create_data_loader(self, cfg, arcface=False, hold_vector=False):
        if arcface or hold_vector:
            normalize = A.Normalize(mean=(0,0,0), std=(1,1,1))
        else:
            normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        val_transform = A.Compose([
                 normalize,
                 ToTensorV2()
                   ])
        if arcface:
            normalize = A.Normalize(mean=(0,0,0), std=(1,1,1))
            val_dst = DataLoader(cfg.X_test, cfg.y_test, transform=val_transform)
        elif hold_vector:
            normalize = A.Normalize(mean=(0,0,0), std=(1,1,1))
            val_dst = DataLoader(cfg.X_train, cfg.y_train, transform=val_transform)
        else:
            val_dst = MetaDataLoader(cfg.X_test, cfg.color_test, cfg.shape_test, transform=val_transform)
        val_loader = data.DataLoader(
                    val_dst, batch_size=1, shuffle=None, num_workers=0, pin_memory=None)
        print(" Query set: %d" %(len(val_dst)))
        return val_loader, val_dst
        
    def load_trained_model(self, cfg, device, weight_path, arcface=False):
        if arcface:
            model = ArcFaceModel(embedding_size=cfg.embedding_size).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            model = RecognitionModel(color_size=cfg.num_color, shape_size=cfg.num_shape).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
        return model
        
    
class Arcface(BasicPredictor):
    def __init__(self, cfg, device, weight_path, hold_vector=False):
        self.num_candidates = 10
        self.model = self.load_trained_model(cfg, device, weight_path, arcface=True)
        if hold_vector:
            self.val_loader, self.val_dst = self.create_data_loader(cfg, arcface=False, hold_vector=True)
            self.create_hold_vector(cfg)
        else:
            self.val_loader, self.val_dst = self.create_data_loader(cfg, arcface=True, hold_vector=False)
            self.predict(cfg)
            
    def load_emmbed_vector(self):
        X_vector = np.load(os.path.join(root, X_embed))
        y_vector = np.load(os.path.join(root, y_embed))
        print(X_vector.shape, y_vector.shape)
        return X_vector, y_vector
        
    def create_hold_vector(self, cfg):
        self.model.eval()
        X, y = [], []
        for i, (x_val, y_val) in tqdm(enumerate(self.val_loader)):
            x_val = x_val.to(device=device)
            label = int(y_val.to('cpu').detach().numpy().copy())
            feature = self.model(x_val)
            feature = feature.to('cpu').detach().numpy().copy()
            X.append(feature)
            y.append(label)
        X, y = np.array(X), np.array(y)
        if len(X.shape)==3:
            X = X.reshape(len(X), X.shape[2])
        print(X.shape, y.shape)
        np.save(os.path.join(root, X_embed), X)
        np.save(os.path.join(root, y_embed), y)

    def predict(self, cfg):
        X_embed, y_embed = self.load_emmbed_vector()
        self.model.eval()
        num_query = len(self.val_dst)
        acc = 0
        for i, (x_val, y_val) in tqdm(enumerate(self.val_loader)):
            x_val = x_val.to(device=device)
            label = int(y_val.to('cpu').detach().numpy().copy())
            feature = self.model(x_val)
            feature = feature.to('cpu').detach().numpy().copy()
            #print(feature.shape, type(feature), label, type(label))
            cos_sims = cosine_similarity(feature, X_embed)
            X_candidate = [y_embed[idx] for idx in np.argsort(cos_sims)[0][::-1][:self.num_candidates]]
            #print(X_candidate, np.argsort(cos_sims)[0][::-1].shape)
            #print(mode(X_candidate), label, mode(X_candidate)==label)
            acc += 1 if mode(X_candidate)==label else 0
        print("total accuracy is ", acc/num_query)


class Recognition(BasicPredictor):
    def __init__(self, cfg, device, weight_path):
        self.model = self.load_trained_model(cfg, device, weight_path, arcface=False)
        self.val_loader, self.val_dst = self.create_data_loader(cfg)
        self.predict(cfg)
        
    def predict(self, cfg):
        self.model.eval()
        num_query = len(self.val_dst)
        sacc, cacc = 0, 0
        for i, (x_img, color, shape) in tqdm(enumerate(self.val_loader)):
            x_img = x_img.to(device=device)
            color = color.to(device=device)
            shape = shape.to(device=device)
            color_pred, shape_pred = self.model(x_img)
            cpred = int(torch.argmax(color_pred).to('cpu').detach().numpy().copy())
            
            spred = int(torch.argmax(shape_pred).to('cpu').detach().numpy().copy())
            
            color = int(color.to('cpu').detach().numpy().copy())
            shape = int(shape.to('cpu').detach().numpy().copy())
            #print(i, shape==shape_pred, color_pred==color)
            sacc += 1 if spred==shape else 0
            cacc += 1 if cpred==color else 0
        print("accuracy color {} shape {}".format(cacc/num_query, sacc/num_query))


class MainPredictor(BasicPredictor):
    def __init__(self, cfg, device, weight_path, meta_weight_path):
        self.cfg = cfg
        self.device = device
        self.num_candidates = 50
        self.arc_model = self.load_trained_model(cfg, device, weight_path=weight_path, arcface=True)
        self.meta_model = self.load_trained_model(cfg, device, weight_path=meta_weight_path, arcface=False)
        self.arc_transform = self.transform_is(arcface=True)
        self.meta_transform = self.transform_is(arcface=False)
        self.predict()
        
    def transform_is(self, arcface=True):
        if arcface:
            normalize = A.Normalize(mean=(0,0,0), std=(1,1,1))
        else:
            normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transform = A.Compose([normalize, ToTensorV2()])
        return transform
        
    def load_data(self, cfg):
        # database
        X, y, color, shape = np.load(cfg.X_test), np.load(cfg.y_test), np.load(cfg.color_test), np.load(cfg.shape_test)
        X_emb, y_emb= np.load(cfg.embed_X), np.load(cfg.embed_y)
        
        print(X.shape, y.shape, color.shape, shape.shape)
        print(X_emb.shape, y_emb.shape)
        return X, y, color, shape, X_emb, y_emb
       
    def numpy2tensor(self, img, transform):
        img = transform(image=img)['image']
        return img
    
    def predict(self):
        X, y, color, shape, X_emb, y_emb = self.load_data(self.cfg)
        print("loading data")
        X_arcface = [self.numpy2tensor(x, transform=self.arc_transform) for x in tqdm(X)]
        X_meta = [self.numpy2tensor(x, transform=self.meta_transform) for x in tqdm(X)]
        self.arc_model.eval()
        self.meta_model.eval()
        
        pred_label = []
        acc, count = 0, 0
        start = time.time()
        for i in tqdm(range(len(X))):
            
            # arcface prediction
            y_test = y[i]
            x_img = X_arcface[i].to(device=device)
            x_img = x_img.unsqueeze(0)
            feature = self.arc_model(x_img)
            feature = feature.to('cpu').detach().numpy().copy()
            cos_sims = cosine_similarity(feature, X_emb)
            candidate_idxes = [y_emb[idx] for idx in np.argsort(cos_sims)[0][::-1][:self.num_candidates]]
             
            # color and shape prediction
            color_pred, shape_pred = self.meta_model(x_img)
            cpred = int(torch.argmax(color_pred).to('cpu').detach().numpy().copy())
            spred = int(torch.argmax(shape_pred).to('cpu').detach().numpy().copy())
            #print(feature.shape, len(candidate_idxes), cpred, spred)

            # color & shape extraction
            for idx in candidate_idxes:
                if dd[str(idx)]["category"] == spred and dd[str(idx)]["color"] == cpred:
                    pred_label.append(idx)
            acc += 1 if y_test in pred_label else 0
            count += 1
        
        print("Inference Latency is", time.time() - start, "[s]")
        print("accuracy is ", acc/count)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-arc', '--arcface', action='store_true', help='arcface prediction')
    parser.add_argument('--hold', action='store_true', help='create hold vector')
    parser.add_argument('--meta', action='store_true', help='meta class recognition')
    parser.add_argument('--main', action='store_true', help='whole class recognition')
    parser.add_argument('-aw', '--arc_weight', type=str, default=None, help='arcface weight path')
    parser.add_argument('-mw', '--meta_weight', type=str, default=None, help='meta model weight path')
    opt = parser.parse_args()
    #if opt.arc_weight is None  opt.meta_weight is None:
    #    print("weight_path should be difined")
    #    sys.exit(1)
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.arcface:
        Arcface(cfg, device, weight_path=opt.arc_weight, hold_vector=False)
    elif opt.hold:
        Arcface(cfg, device, weight_path=opt.arc_weight, hold_vector=True)
    elif opt.meta:
        Recognition(cfg, device, weight_path=opt.meta_weight)
    elif opt.main:
        if opt.arc_weight is None or opt.meta_weight is None:
            print("both model weight_path should be difined")
            sys.exit(1)
        else:
            MainPredictor(cfg, device, weight_path=opt.arc_weight, meta_weight_path=opt.meta_weight)
