import time
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot
import os
from data.dataloader import DataSets, inverse_normalize,preprocess
import torch
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
import matplotlib
matplotlib.use('Agg')

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

def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

path_image_test = 'DataSets/image_test/'
path_image_save = 'DataSets/image_save/'
label_names = list(LABEL_NAMES) + ['bg']

files = os.listdir(path_image_test)
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

checkpoint = torch.load('last_epoch.pkl')
trainer.faster_rcnn.load_state_dict(checkpoint['model_state'])
print(checkpoint['epoch'])
print(checkpoint['optimizer_state']['param_groups'])
opt.caffe_pretrain=False
for f in files:
    print(f)
    img = read_image(path_image_test + f)
    img = torch.from_numpy(img)[None]
    _bboxes, _labels, _scores = faster_rcnn.predict(img,visualize=True)
    
    img = at.tonumpy(img[0])
    bbox = at.tonumpy(_bboxes[0])
    label = at.tonumpy(_labels[0]).reshape(-1)
    score = at.tonumpy(_scores[0]).reshape(-1)


    fig = plot.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
	raise ValueError('The length of score must be same as that of bbox')
    if len(bbox) == 0:
	continue

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],': '.join(caption),style='italic',bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    plot.savefig(path_image_save+f)
print('done')
