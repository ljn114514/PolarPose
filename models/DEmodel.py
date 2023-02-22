import torch, torchvision, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .default import _C as config
from .default import update_config

from .hrnet import get_hrnet
from .kpt_head import KeypointHead

class KeypointModel(nn.Module):
    def __init__(self, 
        num_cpoints=3, 
        width=32,
        feat_channels=256,
        dataset='coco',
        pretrain=None):
        super(KeypointModel, self).__init__()        

        cfg_file = 'models/hrnet/hrnet_w%d_train.yaml'%(width)
        update_config(config, cfg_file)
        self.backbone = get_hrnet(cfg=config, pretrained=pretrain)

        print()
        print('dataset:', dataset)
        print('load pretrained model:', pretrain)
        print('feat_channels', feat_channels)
        self.head = KeypointHead(
            num_cpoints=num_cpoints, 
            in_channels=width*15, 
            feat_channels=feat_channels,
            dataset=dataset)

    def forward(self, img, target=None, img_metas=None, is_train=True, flip_test=False):

        if is_train:
            x = self.backbone(img)
            loss = self.head(x, target, is_train=True)
            return loss

        else:
            if flip_test:
                img_flip = torch.flip(img, dims=[3])
                img = torch.cat([img, img_flip], dim=0)
            x = self.backbone(img)
            results = self.head(x, img_metas=img_metas, is_train=False, flip_test=flip_test)[0]

            pose_result = results[0]
            score = results[1]

            return pose_result, score#, id_preds, pose_pred


def get_model(num_cpoints=3, width=32, feat_channels=64, dataset='coco', pretrained=None):
    
    model = KeypointModel(
        num_cpoints=num_cpoints,
        width=width,
        feat_channels=feat_channels,
        dataset=dataset)

    if pretrained:
        print('load pretrained model:', pretrained)
        weight = torch.load(pretrained, map_location='cpu')
        static = model.state_dict()

        for name, param in weight.items():
            #print('load weight ', name)
            if name.split('.')[0] == 'module':
                name = name[7:]
            if name not in static:
                print('not load weight ', name)
                continue
            #if isinstance(param, nn.Parameter):
            #    param = param.data
            static[name].copy_(param)
            
    return model