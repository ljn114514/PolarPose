import os, cv2
import random, time, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.backends.cudnn as cudnn

#from dataset.PoseDataset import PoseDataset, batch_collate
from dataset.COCOPoseDataset_radius import COCOPoseDataset, batch_collate
import models.DEmodel
from utils import lr_adjust
from utils.logger import Logger

sys.stdout = Logger()
##########   PARAMETER ###########
device_id = torch.device('cuda', 0)
#device_id = torch.device('cpu')
base_lr = 1e-3
total_epoch = 75
lr_step = [61, 71]
#lr_step = [126, 136]

batch_size = 18
input_size = 512
width = 32
feat_channel = 256
##########   DATASET   ###########
train_dataset = COCOPoseDataset(
	ann_file='dataset/coco/annotations/person_keypoints_train2017.json',
	img_prefix='dataset/coco/train2017',
	input_size=512,
	test_mode=False)
train_loader = torch.utils.data.DataLoader(
	train_dataset, 
	batch_size=batch_size, 
	shuffle=True, 
	num_workers=4, 
	collate_fn=batch_collate)

##########   MODEL   ###########
model = models.DEmodel.KeypointModel(
	num_cpoints=3, width=width,
	feat_channels=feat_channel,
	dataset='coco',
	pretrain='pretrain/hrnetv2_w32_imagenet_pretrained.pth')
# model = models.DEmodel.get_model(
#     num_cpoints=3, width=width,
#     feat_channels=feat_channel,
#     dataset='coco',
#     pretrained='weight/HRNet32_pose_00060.pth')
model.to(device_id)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

##########   TRAIN   ###########
for epoch in range(1, total_epoch+1):

	lr = lr_adjust.StepLrUpdater(epoch, base_lr=base_lr, gamma=0.1, step=lr_step)
	lr_adjust.SetLr(lr, optimizer)
	
	running_loss_cls = 0.
	running_loss_ang = 0.
	running_loss_len = 0.

	model.train()
	torch.cuda.empty_cache()
	for i, data in enumerate(train_loader, 1):
		#break
		images, gt_heatmaps, inst_coords, inst_heatmaps, idxs, masks = data
		images = images.to(device_id)
		gt_heatmaps = gt_heatmaps.to(device_id)
		masks = masks.to(device_id)

		#print(i, images.size(), gt_heatmaps.size(), masks.size())

		loss_cls, loss_ang, loss_len = model(images, target=(gt_heatmaps, inst_coords, inst_heatmaps, masks))
		losses = loss_cls.mean()+loss_ang.mean()+loss_len.mean()

		optimizer.zero_grad()
		losses.backward()
		optimizer.step()
		
		running_loss_cls = running_loss_cls + loss_cls.mean()
		running_loss_ang = running_loss_ang + loss_ang.mean()
		running_loss_len = running_loss_len + loss_len.mean()

		if i % 300 == 0:
			print('[%d/%d] iter: %d/%d. lr:%f. loss_cls: %.3f, loss_ang:%.3f, loss_len:%.3f'%
				(epoch, total_epoch, i, len(train_loader), lr, 
					running_loss_cls/i, 
					running_loss_ang/i, 
					running_loss_len/i))

	print('Finish %d epoch'%(epoch))

	save_path = 'weight/HRNet%d_pose_%05d.pth'%(width, epoch)
	if hasattr(model, 'module'):
		torch.save(model.module.state_dict(), save_path)
	else:
		torch.save(model.state_dict(), save_path)
	#break