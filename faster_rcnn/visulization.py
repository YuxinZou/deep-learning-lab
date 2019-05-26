#%load_ext autoreload
#%autoreload 2
import os
from data.dataloader import DataSets, inverse_normalize #TestDataset,
import numpy as np
import torch
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot
from data.dataloader import preprocess
#%matplotlib inline
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
label_names = list(LABEL_NAMES) + ['bg']
path = 'DataSets/image_test/'
files = os.listdir(path)
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

checkpoint = torch.load('model.pkl')
trainer.faster_rcnn.load_state_dict(checkpoint['model_state'])
print(checkpoint['epoch'])
print(checkpoint['optimizer_state']['param_groups'])
opt.caffe_pretrain=False
for f in files:
    print(f)
    name_list = list()
    img = read_image(path + f)
    #img = preprocess(img)
    img = torch.from_numpy(img)[None]
    #print(img,img.shape)
    _bboxes, _labels, _scores = faster_rcnn.predict(img,visualize=True)
    #if len(_labels) != 0:
    #    for label in _labels:
    #        name_list.append(label_names[label])
    print(_bboxes)
    print(_labels)
    print(_scores)
    ax = vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
"""
image_folder_path = 'DataSets/images/'
cvs_file_path = 'DataSets/labels.csv'

dataset = DataSets(cvs_file_path,image_folder_path)
data_size = len(dataset)
indices = list(range(data_size))
split = int(np.floor(data_size * 0.02))
np.random.seed(42)
np.random.shuffle(indices)
train_indices,val_indices = indices[split:],indices[:split]
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset,batch_size=1,sampler = train_sampler)
val_loader = torch.utils.data.DataLoader(dataset,batch_size=1,sampler = valid_sampler)
print('load data')
for ii, (_,imgs, sizes, gt_bboxes_, gt_labels_,_, gt_difficults_) in enumerate(val_loader):
    sizes = [sizes[0][0].item(), sizes[1][0].item()]
    pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
    print(pred_bboxes_, pred_labels_, pred_scores_)
"""
# it failed to find the dog, but if you set threshold from 0.7 to 0.6, you'll find it

#fig = plt.gcf()
    #fig.set_size_inches(11, 5)
    #fig.savefig(image_save + 'test_'+str(batch_idx)+'.jpg', dpi=100)
