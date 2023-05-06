import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import json 
import glob
from .utils import load_json, img_padding

root = 'dataset/train'
nppath = 'dataset/npy'
meta_data = load_json("dataset/cs_label.json")
print(meta_data)

def preprocess(p, size=260, use_train=True):
    img = cv2.imread(p)
    if use_train:
        img = img_padding(img, desired_size=size)
    else:
        img = img_padding(img, desired_size=size)
        img = np.fliplr(img)
    return img

def Tonumpy(X, y, shape, color, fname):
    X, y, shape, color = np.array(X), np.array(y), np.array(shape), np.array(color)
    print(X.shape, y.shape, shape.shape, color.shape)
    np.save(os.path.join(nppath, "X_{}".format(fname)), X)
    np.save(os.path.join(nppath, "cls_{}".format(fname)), y)
    np.save(os.path.join(nppath, "shape_{}".format(fname)), shape)
    np.save(os.path.join(nppath, "color_{}".format(fname)), color)

def prepare(use_train=True):
    os.makedirs(nppath, exist_ok=True)
    labels = os.listdir(root)
    labels = sorted(labels)
    imgs, cls, shape, color = [], [], [], []
    for i, label in enumerate(labels):
        dir_path = os.path.join(root, label, '*.jpg')
        print(len(labels)-1, i, label, dir_path)
        if i > len(labels)-1:
            break
        for im_path in glob.glob(dir_path):
            img = preprocess(im_path, use_train=use_train)
            if meta_data[str(i)]:
                shape_label = meta_data[str(i)]['category']
                color_label = meta_data[str(i)]['color']
                #print(i, im_path, img.min(), img.max(), img.shape)
                imgs.append(img)
                cls.append(i)
                shape.append(shape_label)
                color.append(color_label)
    fname = 'train' if use_train else 'test'
    Tonumpy(imgs, cls, shape, color, fname)

