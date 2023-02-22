import numpy as np
import torch

def meta_convert(img_num, img_meta=None):
    img_meta_new = []
    for i in range(img_num):
        img_meta_new_item = {}

        ori_shape = img_meta['ori_shape']
        img_meta_new_item['ori_shape']=[ori_shape[0][i].item(), ori_shape[1][i].item(), ori_shape[2][i].item()]

        img_shape = img_meta['img_shape']
        img_meta_new_item['img_shape']=[img_shape[0][i].item(), img_shape[1][i].item(), img_shape[2][i].item()]

        pad_shape = img_meta['pad_shape']
        img_meta_new_item['pad_shape']=[pad_shape[0][i].item(), pad_shape[1][i].item(), pad_shape[2][i].item()]

        scale_factor = img_meta['scale_factor']
        img_meta_new_item['scale_factor']=scale_factor[i]#.numpy()

        flip = img_meta['flip']
        img_meta_new_item['flip']=flip[i].item()

        img_meta_new.append(img_meta_new_item)


    #print('img_meta_new', img_meta_new)
    return img_meta_new
    #a