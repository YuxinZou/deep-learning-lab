from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from tqdm import tqdm
import numpy as np
import torch
from utils.config import opt
from data.dataloader import DataSets, inverse_normalize #TestDataset, 
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from tensorboardX import SummaryWriter
from torchnet.meter import AverageValueMeter, MovingAverageValueMeter
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (_,imgs, sizes, gt_bboxes_, gt_labels_,_, gt_difficults_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        print(pred_bboxes_, pred_labels_, pred_scores_)
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
        print(ii,'done')

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    image_folder_path = 'DataSets/images/'
    cvs_file_path = 'DataSets/labels.csv'

    dataset = DataSets(cvs_file_path,image_folder_path)
    data_size = len(dataset) 
    indices = list(range(data_size))
    split = int(np.floor(data_size * 0.2))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices,val_indices = indices[split:],indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
	    
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=1,sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,batch_size=1,sampler = valid_sampler)
    print('load data')

    avg_loss = AverageValueMeter()
    ma20_loss = MovingAverageValueMeter(windowsize=20)
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    start_epoch = 0
    best_map = -100
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
	checkpoint = torch.load(opt.load_path)
	start_epoch = checkpoint['epoch']
	best_map = checkpoint['best_map']
    	trainer.faster_rcnn.load_state_dict(checkpoint['model_state']) 
        print("> Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))

    #trainer.vis.text(dataset.db.label_names, win='labels')

 # set tensor-board for visualization
    writer = SummaryWriter('runs/'+opt.log_root)


    lr_ = opt.lr
    for epoch in range(start_epoch,opt.epoch):
        trainer.reset_meters()
        for ii, (img,_,_, bbox_, label_, scale,_) in enumerate(train_loader):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            loss  = trainer.train_step(img, bbox, label, scale)
            #print(loss)
            #print(loss.total_loss)
            loss_value = loss.total_loss.cpu().data.numpy()
            avg_loss.add(float(loss_value))
            ma20_loss.add(float(loss_value))
	    print('[epoch:{}/{}]  [batch:{}/{}]  [sample_loss:{:.4f}] [avg_loss:{:.4f}]  [ma20_loss:{:.4f}]'.format(epoch,opt.epoch, ii+1, len(train_loader), loss.total_loss.data,avg_loss.value()[0], ma20_loss.value()[0]))
		

            if (ii + 1) % opt.plot_every == 0:
	        niter = epoch*len(train_loader)+ii
                writer.add_scalar('Train/Loss', ma20_loss.value()[0], niter)


        eval_result = eval(val_loader, faster_rcnn, test_num=opt.test_num)
	print(eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            state = {"epoch": epoch + 1,
                     "best_map": best_map,
                     "model_state": trainer.faster_rcnn.state_dict()}
                    # "optimizer_state": optimizer.state_dict()}
            torch.save(state, opt.model_para)
        if epoch == 9:
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

    writer.close()

if __name__ == '__main__':
    train()
