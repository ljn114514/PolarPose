import os, cv2
import random, time, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import torch

import models.DEmodel
from dataset.COCOPoseDataset_radius import COCOPoseDataset, batch_collate_test
from utils.coco_eval.coco_eval import CocoEvaluator

##########   PARAMETER   ###########
device_id = torch.device('cuda', 0)
#device_id = torch.device('cpu')
width = 32
feat_channel = 256
##########   DATASET   ###########
eval_dataset = COCOPoseDataset(
    ann_file='dataset/coco/annotations/person_keypoints_val2017.json',
    img_prefix='dataset/coco/val2017',
    #img_scale_test=[(10000, 384), (10000, 512), (10000, 768), (10000, 1024), (10000, 1280), (10000, 1408)],
    img_scale_test=(3000, 512),
    test_mode=True)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2, 
    collate_fn=batch_collate_test)
###### eval ########
coco = eval_dataset.coco#COCO(ann_file)
coco_evaluator = CocoEvaluator(coco, ['keypoints'])

snapshot = 'weight/HRNet32_pose_00140.pth'
pose_model = models.DEmodel.get_model(
    num_cpoints=3, width=width,
    feat_channels=feat_channel,
    dataset='coco',
    pretrained=snapshot)

pose_model.to(device_id)
pose_model.eval()

################## eval ##################
time_count = []
for i, data in enumerate(eval_loader, 0):
    #print()

    if i%1000 ==0: print('num', i, len(eval_dataset))
    imgs, img_metas, idx = data
    #print(img_metas)

    im_index = eval_dataset.img_ids[idx[0]]
    
    imgs = imgs.to(device_id)
    #torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        pose_results, scores = pose_model(imgs, img_metas=img_metas, is_train=False, flip_test=True)
    #torch.cuda.synchronize()        
    time_cost = time.time() - start_time
    time_count.append(time_cost)

    if pose_results.size(0) == 0: continue


    a = pose_results
    bbox_results = torch.stack([a[:, :, 0].min(1)[0], a[:, :, 1].min(1)[0], a[:,:, 0].max(1)[0], a[:,:, 1].max(1)[0]],-1)
    bbox_results = torch.cat([bbox_results, scores.unsqueeze(dim=1)], dim=1)

    outputs = {'boxes': bbox_results.cpu()[:,:4],
               'labels':torch.ones(pose_results.size(0)),
               'scores':scores.cpu(),
               'keypoints':pose_results[:,:,:3].cpu(),
               'keypoints_scores':pose_results[:,:,-1].cpu()}

    res = {im_index:outputs}
    coco_evaluator.update(res)

    if len(time_count) > 100: time_count = time_count[1:]
    print(i, 'avg time:', torch.tensor(time_count).mean().item(), snapshot)
    if i> 10000: break

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()
