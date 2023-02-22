import numpy as np
import copy, torch, sys



def computeOks(gt_pose, dt_pose):

    #sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79])/10.0
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    var = (sigmas * 2)**2
    var = torch.tensor(var, device=gt_pose.device).float()

    gts = gt_pose
    dts = dt_pose

    xg, yg, vg = gts[:, :, 0], gts[:, :, 1], gts[:, :, 2]
    xd, yd, vd = dts[:, :, 0], dts[:, :, 1], dts[:, :, 2]

    bbox_gt = torch.stack([xg.min(dim=1)[0], yg.min(dim=1)[0], xg.max(dim=1)[0], yg.max(dim=1)[0]],-1)
    bbox_dt = torch.stack([xd.min(dim=1)[0], yd.min(dim=1)[0], xd.max(dim=1)[0], yd.max(dim=1)[0]],-1)
    bbox_area_gt = (bbox_gt[:, 2]-bbox_gt[:, 0])*(bbox_gt[:, 3]-bbox_gt[:, 1])
    bbox_area_dt = (bbox_dt[:, 2]-bbox_dt[:, 0])*(bbox_dt[:, 3]-bbox_dt[:, 1])

    xg = xg.unsqueeze(dim=1)*torch.ones((1,xd.size(0),1), device=xg.device)
    yg = yg.unsqueeze(dim=1)*torch.ones((1,yd.size(0),1), device=yg.device)
    xd = xd.unsqueeze(dim=0)*torch.ones((xg.size(0),1,1), device=xd.device)
    yd = yd.unsqueeze(dim=0)*torch.ones((yg.size(0),1,1), device=yd.device)

    dx = xd - xg
    dy = yd - yg

    bbox_area = bbox_area_gt.unsqueeze(dim=1)*bbox_area_dt.unsqueeze(dim=0)
    bbox_area = bbox_area.sqrt().unsqueeze(dim=2).clamp(min=0.01)

    #bbox_area_gt = bbox_area_gt.unsqueeze(dim=1)*torch.ones((1,bbox_area_dt.size(0)), device=bbox_area_gt.device)
    #bbox_area_dt = bbox_area_dt.unsqueeze(dim=0)*torch.ones((bbox_area_gt.size(0),1), device=bbox_area_gt.device)
    #bbox_area = torch.cat([bbox_area_dt.unsqueeze(dim=0), bbox_area_gt.unsqueeze(dim=0)], dim=0).min(dim=0)[0].unsqueeze(dim=2)

    var_temp = var.unsqueeze(dim=1).unsqueeze(dim=2)
    e = -(dx**2 + dy**2) / var / bbox_area / 2
    ious = e.exp().mean(dim=2) 

    #vis = vg.unsqueeze(dim=1)*vd.unsqueeze(dim=0)
    #ious = (e.exp()*vis).sum(dim=2)/vis.sum(dim=2).clamp(min=0.01)

    #print('e', e.size(), gt_pose.size(), dt_pose.size())
    #print('vg, vd', vg.size(), vd.size())
    #print()
    return ious


def oks_nms(kpts, scores, sigmas=None, in_vis_thre=None, thresh=0.5):
    #print('oks_nms', oks_nms)
    order = torch.sort(scores, descending=True)[1]
    oks = computeOks(kpts, kpts)

    keep = []
    while order.size(0) > 0:
        i = order[0]
        keep.append(i)
        oks_ovr = oks[i, order[1:]]

        inds = torch.nonzero(oks_ovr <= thresh).view(-1)
        order = order[inds + 1]

    return torch.tensor(keep)




# np_dir = '/home/ljn/disk1/PolarPose_final_ver/PolarPose_test_crowdpose_res50_300epoch/result_nms0.6/'
# gt_pose = np.load(np_dir+'100005_pose.npy')
# dt_pose = np.load(np_dir+'100025_pose.npy')
# gt_pose = torch.tensor(gt_pose)
# dt_pose = torch.tensor(dt_pose)
# ious = computeOks(gt_pose, dt_pose, var)