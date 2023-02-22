import os, cv2
import random, time, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.ddp_util import find_free_port, setup, cleanup

#from dataset.COCOPoseDataset import COCOPoseDataset, batch_collate
#from dataset.PoseDataset import PoseDataset, batch_collate
from dataset.COCOPoseDataset_radius import COCOPoseDataset, batch_collate
import models.DEmodel
from utils import lr_adjust
from utils.logger import Logger

#sys.stdout = Logger('./log_train.txt')
##########   PARAMETER ###########
#device_id = torch.device('cuda', 0)
base_lr = 1e-3
total_epoch = 140#75
lr_step = [126, 136]#[61, 71]

batch_size=30
world_size = 2 # GPU num

input_size = 512
width = 32
feat_channel = 256
pretrain='pretrain/hrnetv2_w32_imagenet_pretrained.pth'
##########   TRAIN#### ###########
def main_worker(local_rank, world_size, port):
	"""
	gpu: the local rank in the current node. Should consider all nodes to compute the global rank for initializing the process_group
	"""

	#rank = cfg.node_no * cfg.gpus_per_node + gpu
	rank = local_rank
	setup(rank, world_size, port)  # perform setup within each process before instantiating DDP
	torch.cuda.set_device(local_rank)

	if rank == 0: sys.stdout = Logger()

	##########   DATASET   ###########
	train_dataset = COCOPoseDataset(
		ann_file='dataset/coco/annotations/person_keypoints_train2017.json',
		img_prefix='dataset/coco/train2017',
		input_size=input_size,
		test_mode=False)
	sampler = DistributedSampler(train_dataset, rank=rank)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, 
		batch_size=batch_size//world_size, 
		shuffle=False, 
		num_workers=4,
		sampler=sampler, 
		collate_fn=batch_collate)

	##########   MODEL   ###########
	# model = models.DEmodel.KeypointModel(
	# 	num_cpoints=3, width=width,
	# 	feat_channels=feat_channel,
	# 	dataset='coco',
	# 	pretrain=pretrain)
	model = models.DEmodel.get_model(
	    num_cpoints=3, width=width,
	    feat_channels=feat_channel,
	    dataset='coco',
	    pretrained='weight/HRNet32_pose_00060.pth')
	model.cuda()
	model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
	optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

	##########   TRAIN   ###########
	for epoch in range(61, total_epoch+1):

		lr = lr_adjust.StepLrUpdater(epoch, base_lr=base_lr, gamma=0.1, step=lr_step)
		lr_adjust.SetLr(lr, optimizer)
		
		running_loss_cls = 0.
		running_loss_ang = 0.
		running_loss_len = 0.

		sampler.set_epoch(epoch)

		model.train()
		torch.cuda.empty_cache()
		for i, data in enumerate(train_loader, 1):

			images, gt_heatmaps, inst_coords, inst_heatmaps, idxs, masks = data
			images = images.cuda()
			gt_heatmaps = gt_heatmaps.cuda()
			masks = masks.cuda()


			loss_cls, loss_ang, loss_len = model(images, target=(gt_heatmaps, inst_coords, inst_heatmaps, masks))

			losses = loss_cls.mean()+loss_ang.mean()+loss_len.mean()

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()
			
			running_loss_cls = running_loss_cls + loss_cls.mean()
			running_loss_ang = running_loss_ang + loss_ang.mean()
			running_loss_len = running_loss_len + loss_len.mean()

			if i % 300 == 0 and rank == 0:
				print('[%d/%d] iter: %d/%d. lr:%f. loss_cls: %.3f, loss_ang:%.3f, loss_len:%.3f'%
					(epoch, total_epoch, i, len(train_loader), lr, 
						running_loss_cls/i, 
						running_loss_ang/i, 
						running_loss_len/i))

		if rank == 0:
			print('Finish %d epoch'%(epoch))
			save_path = 'weight/HRNet%d_pose_%05d.pth'%(width, epoch)
			if hasattr(model, 'module'):
				torch.save(model.module.state_dict(), save_path)
			else:
				torch.save(model.state_dict(), save_path)


	i = 0
	while 1:
		time.sleep(10)
		i += 1
		print(i)		
	cleanup()

def run_demo(demo_fn, world_size):
	port = find_free_port()
	print('world_size', world_size, 'port', port)
	mp.spawn(demo_fn,
			 args=(world_size, port),  # the parameters to demo_fn
			 nprocs=world_size,
			 join=True)
# world_size: total number of processes, equal to GPU number, = node * gpu_per_node
# node: usually refer to the number of machines, each machine may have multiple GPUs
if __name__ == '__main__':
	run_demo(main_worker, world_size)