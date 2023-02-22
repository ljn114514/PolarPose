import numpy as np
import torch


def oks_nms(poses, scores, thresh, sigmas=None, in_vis_thre=None):
    if len(poses) == 0: return []
    areas = (np.max(poses[:, :, 0], axis=1) - np.min(poses[:, :, 0], axis=1)) * \
            (np.max(poses[:, :, 1], axis=1) - np.min(poses[:, :, 1], axis=1))
    poses = poses.reshape(poses.shape[0], -1)

    order = scores.argsort()[::-1]

    keep = []
    keep_ind = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        oks_ovr = oks_iou(poses[i], poses[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)
        inds = np.where(oks_ovr <= thresh)[0]
        nms_inds = np.where(oks_ovr > thresh)[0]
        nms_inds = order[nms_inds + 1]
        keep_ind.append(nms_inds.tolist())
        order = order[inds + 1]

    return keep, keep_ind

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    vars = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg >= in_vis_thre) and list(vd >= in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def computeOks(gt_pose, dt_pose, sigmas=None, area=None):

    var = (sigmas * 2)**2
    var = torch.tensor(var, device=gt_pose.device).float()

    gts = gt_pose
    dts = dt_pose

    xg, yg, vg = gts[:, :, 0], gts[:, :, 1], gts[:, :, 2]
    xd, yd, vd = dts[:, :, 0], dts[:, :, 1], dts[:, :, 2]

    bbox_gt = torch.stack([xg.min(dim=1)[0], yg.min(dim=1)[0], xg.max(dim=1)[0], yg.max(dim=1)[0]],-1)
    bbox_dt = torch.stack([xd.min(dim=1)[0], yd.min(dim=1)[0], xd.max(dim=1)[0], yd.max(dim=1)[0]],-1)    

    xg = xg.unsqueeze(dim=1)*torch.ones((1,xd.size(0),1), device=xg.device)
    yg = yg.unsqueeze(dim=1)*torch.ones((1,yd.size(0),1), device=yg.device)
    xd = xd.unsqueeze(dim=0)*torch.ones((xg.size(0),1,1), device=xd.device)
    yd = yd.unsqueeze(dim=0)*torch.ones((yg.size(0),1,1), device=yd.device)

    dx = xd - xg
    dy = yd - yg
    
    if area is not None:
        area = area.unsqueeze(dim=1).unsqueeze(dim=1)
        area = area*torch.ones(1, dt_pose.size(0), 1)
    else:
        bbox_area_gt = (bbox_gt[:, 2]-bbox_gt[:, 0])*(bbox_gt[:, 3]-bbox_gt[:, 1])
        bbox_area_dt = (bbox_dt[:, 2]-bbox_dt[:, 0])*(bbox_dt[:, 3]-bbox_dt[:, 1])
        bbox_area = bbox_area_gt.unsqueeze(dim=1)*bbox_area_dt.unsqueeze(dim=0)
        area = bbox_area.sqrt().unsqueeze(dim=2).clamp(min=0.01)


    var_temp = var.unsqueeze(dim=1).unsqueeze(dim=2)
    #e = -(dx**2 + dy**2) / var / bbox_area / 2
    e = -(dx**2 + dy**2) / var / area / 2
    ious = e.exp().mean(dim=2) 

    return ious