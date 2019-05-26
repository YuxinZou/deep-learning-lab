from __future__ import  absolute_import
from __future__ import  division
import sys
sys.path.append("..")
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt
import os
from torch.utils import data
from .util import read_image

import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from PIL import Image
import csv

import pandas as pd

def label_reorder(path):
    data = pd.read_csv(path)
    set_class = set(data['label'])
    set_image = set(data['image_path'])
    aa = []
    for image in set_image:   
        subdata = data[data['image_path'] == image]
        a = []
        a.append(image)
        #a.append(subdata['image_width'][0])
        #a.append(subdata['image_height'][0])
        for index,row in subdata.iterrows():
            a.append(row['xmin'])
            a.append(row['ymin'])
            a.append(row['xmax'])
            a.append(row['ymax'])
            a.append(row['label'])
        aa.append(a)
    return aa

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.5912, 0.5749, 0.5676],std=[0.2575, 0.3002, 0.2340])
            #mean=[0.485, 0.456, 0.406],
            #                    std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img).float())
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

def get_list(path):
    with open(path,'rb') as myfile:
        reader = csv.reader(myfile)
        return list(reader)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

class DataSets(data.Dataset):
    def __init__(self, cvs_file_path,image_folder_path,
                 use_difficult=False, return_difficult=False,
                 ):
	self.data = label_reorder(cvs_file_path) # e.g. ['14373926_1777422745865259_4917968696174968832_n.jpg', 358, 468, 806, 952, 'Top', 862, 808, 1080, 1163, 'Handbag', 295, 450, 913, 961, 'Tee']  totally 1863 images
	#self.data = get_list(cvs_file_path)
        self.image_training_folder_path = image_folder_path
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = LABEL_NAMES
	self.num_img = len(self.data)
	self.tsf = Transform(opt.min_size, opt.max_size)

    def get_example(self, index):
	subdata = self.data[index]
	box=[]
	label=[]
	difficult =[]
	num_boxes = (len(subdata) - 1) // 5
	for i in range(num_boxes):
	    xmin = float(subdata[1+5*i])
            ymin = float(subdata[2+5*i])
            xmax = float(subdata[3+5*i])
	    ymax = float(subdata[4+5*i])
	    c = LABEL_NAMES.index(subdata[5+5*i])
	    box.append([ymin,xmin,ymax,xmax])
	    label.append(int(c))
	    difficult.append(0)
	box = np.array(box).astype(np.float32)
	label = np.array(label).astype(np.float32)
	difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        # Load a image
	image_name = subdata[0]
	print(image_name)
        img = read_image(self.image_training_folder_path + image_name, color=True)
        img_val = preprocess(img)
	img_train, bbox, label, scale = self.tsf((img, box, label))
        return img_train.copy(),img_val.copy(),img.shape[1:], bbox.copy(), label.copy(), scale,difficult.copy()

    __getitem__ = get_example

    def __len__(self):
        return self.num_img


LABEL_NAMES = (
	'Blouse',
	'Sweatpants', 
	'Cardigan', 
	'Shirt', 
	'Brassiere', 
	'Luggage and bags', 
	'Top', 
	'Suit', 
	'Swimwear', 
	'_Underwear', 
	'High heels', 
	'Trousers', 
	'Tank', 
	'Outter', 
	'Sunglasses', 
	'Shorts', 
	'Leggings', 
	'Hoodie', 
	'Handbag', 
	'Sweater', 
	'Blazer', 
	'Coat', 
	'Jacket', 
	'Footwear', 
	'Sweatshorts', 
	'Romper', 
	'Scarf', 
	'Headwear', 
	'Dress', 
	'Jeans', 
	'Tee', 
	'Jumpsuit', 
	'Skirt', 
	'Belt')
if __name__ == "__main__":
    image_folder_path = '../DataSets/images/'
    cvs_file_path = '../DataSets/test.csv'
    a = DataSets(cvs_file_path,image_folder_path)
    aa,bb,cc,dd = a.__getitem__(0)
    print(aa,bb,cc,dd)
