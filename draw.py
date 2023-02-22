import os, cv2
import random, time, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='2'

import torch

import models.DEmodel
from dataset.COCOPoseDataset import COCOPoseDataset, batch_collate_test
from utils.coco_eval.coco_eval import CocoEvaluator

##########   PARAMETER   ###########
device_id = torch.device('cuda', 0)
#device_id = torch.device('cpu')
width = 32
feat_channel = 64

colors = [[  0,  0,255], [  0,255,  0], [255,0,0], 
          [255,255,  0], [255,  0,255], [  0,255,255],
          [  0, 170, 255], [0,255,170], [170,255,0],
          [170,0,255], [255,170,0], [255,0,170],
          [170, 0, 0], [0, 170, 0], [0,0,170],
          [170, 170, 0], [170, 0, 170], [0, 170, 170], [255, 255, 255]]

lines = [[0,2], [2,4], [1,3], [3,5], [0,6], [1,7], [6,8], [8,10], [7,9], [9,11], [0,13], [1,13], [13,12]]
lines = [[0,1], [0,2], [1,3], [2,4], 
         [0,5], [0,6], [5,7], [6,8], [7,9], [8,10], 
         [5,11], [6,12], [11,13], [12,14], [13,15], [14,16]]


##########   DATASET   ###########
eval_dataset = COCOPoseDataset(
    ann_file='dataset/coco/annotations/person_keypoints_val2017.json',
    img_prefix='dataset/coco/val2017',
    #img_scale_test=[(10000, 384), (10000, 512), (10000, 768), (10000, 1024), (10000, 1280), (10000, 1408)],
    img_scale_test=(2000, 512),
    test_mode=True)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=0, 
    collate_fn=batch_collate_test)
###### eval ########
coco = eval_dataset.coco#COCO(ann_file)
coco_evaluator = CocoEvaluator(coco, ['keypoints'])

snapshot = 'weight/HRNet32_pose_00009.pth'
pose_model = models.DEmodel.get_model(
    num_cpoints=3, width=width,
    feat_channels=feat_channel,
    dataset='coco',
    pretrained=snapshot)

pose_model.to(device_id)
pose_model.eval()

################## eval ##################
time_count = []
bbox_score_thr = 0.1
img_prefix='dataset/coco/val2017/'
for i, data in enumerate(eval_loader, 0):
    #print()

    #img_info = eval_dataset.img_infos[i]
    img_id = eval_dataset.img_ids[i]
    img_name = eval_dataset.coco.loadImgs(img_id)[0]['file_name']
    img = cv2.imread(img_prefix+ img_name)

    if i%1000 ==0: print('num', i, len(eval_dataset))
    imgs, img_metas, idx = data
    #print(img_metas)

    im_index = eval_dataset.img_ids[idx[0]]
    
    imgs = imgs.to(device_id)
    #torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        pose_results, scores = pose_model(imgs, img_metas=img_metas, is_train=False, flip_test=False)
    #torch.cuda.synchronize()        
    time_cost = time.time() - start_time
    time_count.append(time_cost)

    if pose_results.size(0) == 0: continue

    print('images', i, scores.shape, pose_results.shape)
    print()

    color = (255,255,255)
    inds = torch.nonzero(torch.tensor(scores)>bbox_score_thr).view(-1)
    if inds.size(0)<5:
        continue

    for i in inds:
        #bbox = bbox_results[i]
        pose = pose_results[i]

        #bbox = tuple(x for x in bbox)
        #cv2.rectangle(img, bbox[0:2], bbox[2:4], color, 2)

        for line in lines:
            start_point = (pose_results[i, line[0], 0], pose_results[i, line[0], 1])
            end_point  =  (pose_results[i, line[1], 0], pose_results[i, line[1], 1])
            cv2.line(img, start_point, end_point, color, 2)

        for k in range(pose_results.shape[1]):
            if pose_results[i, k, 2] >0:
                point = pose_results[i, k, 0:2]
                cv2.circle(img, (point[0],point[1]), 2, colors[i%len(colors)], 2)

    cv2.imshow('1', img)
    cv2.waitKey(0)
    