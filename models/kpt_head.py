import torch, cv2, math, time, torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .conv_module import ConvModule 

#from .weight_init import bias_init_with_prob, normal_init
from .loss import Neg_loss as FocalLoss
from .nms import oks_nms, oks_iou, computeOks

class KeypointHead(nn.Module):

	def __init__(self,
				 num_cpoints,
				 in_channels=480,
				 feat_channels=256,
				 dataset='coco'):

		super(KeypointHead, self).__init__()		

		if dataset == 'coco': #COCO
			self.num_keypoints = 17
			self.flip_index = [0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15]
			sigmas = [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89]
			self.sigmas = np.array(sigmas) / 10.0
		else: #CrowdPose
			self.num_keypoints = 14
			self.flip_index = [1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 12, 13]
			self.sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79])/10.0
			#self.sigmas = np.array(sigmas) / 10.0

		#self.loss_weight = 3
		print('num_keypoints:', self.num_keypoints)
		print('num_cpoints:', num_cpoints)
		#print('loss_weight', self.loss_weight)

		self.num_cpoints = num_cpoints
		self.cls_out_channels = self.num_keypoints+self.num_cpoints

		self.in_channels = in_channels
		self.feat_channels = feat_channels
		
		self.angle_bins = 90

		self.loss_cls = FocalLoss()
		self._init_layers()

		#self.temperature = 0.02
		self.max_person_per_img = 30
		#print('temperature', self.temperature)
		
		self.nms_pre = 50
		self.score_thr = 0.05
		self.max_per_img = 50
		self.nms_thr = 0.5

	def _init_layers(self):

		self.cls_convs = nn.ModuleList()
		self.len_convs = nn.ModuleList()
		self.ang_convs = nn.ModuleList()

		for i in range(2):
			chn = self.in_channels if i == 0 else self.feat_channels
			self.cls_convs.append(
				ConvModule(chn, self.feat_channels, kernel_size=3, stride=1, padding=1))
			self.len_convs.append(
				ConvModule(chn, self.feat_channels, kernel_size=3, stride=1, padding=1))
			self.ang_convs.append(
				ConvModule(chn, self.feat_channels, kernel_size=3, stride=1, padding=1))

		for m in self.cls_convs:
			nn.init.normal_(m.conv.weight, mean=0, std=0.01)
		for m in self.len_convs:
			nn.init.normal_(m.conv.weight, mean=0, std=0.01)
		for m in self.ang_convs:
			nn.init.normal_(m.conv.weight, mean=0, std=0.01)

		self.cls_head = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1, padding=0)
		self.ang_head = nn.Linear(self.feat_channels, self.num_keypoints*self.angle_bins)
		self.len_head = nn.Linear(self.feat_channels, self.num_keypoints)

		prior_prob = 0.01
		bias_value = -math.log((1 - prior_prob) / prior_prob)
		torch.nn.init.normal_(self.cls_head.weight, std=0.001)		
		torch.nn.init.constant_(self.cls_head.bias, bias_value)

		torch.nn.init.normal_(self.ang_head.weight, std=0.001)
		torch.nn.init.constant_(self.ang_head.bias, 0)

		torch.nn.init.normal_(self.len_head.weight, std=0.001)
		torch.nn.init.constant_(self.len_head.bias, 0)

	def forward(self, x, targets=None, img_metas=None, is_train=True, flip_test=False):
		#print('x', x.size())
		if is_train:
			outputs = self.forward_train(x, targets)
		else:
			if flip_test:
				outputs = self.forward_test_flip(x, img_metas)
			else:
				outputs = self.forward_test(x, img_metas)

		return outputs

	def forward_train(self, x, targets):
		#print('\n')

		img_num = x.size(0)
		max_person_num = img_num*self.max_person_per_img


		heatmaps, inst_coords, quantized_len_angles, masks = targets

		cls_feat = x
		len_feat = x
		ang_feat = x

		for cls_layer in self.cls_convs:
			cls_feat = cls_layer(cls_feat)
		cls_scores = self.cls_head(cls_feat)

		for len_layer in self.len_convs:
			len_feat = len_layer(len_feat)
		#pred_lens = self.len_head(len_feat)

		for ang_layer in self.ang_convs:
			ang_feat = ang_layer(ang_feat)
		

		#cls_scores = self.cls_head(x)
		#ang_feat = self.pose_angle(x)
		#len_feat = self.pose_length(x)

		gt_len_angles = torch.cat(quantized_len_angles, dim=0).to(cls_scores.device)
		#pred_angs, pred_lens = [], []
		num_gts = 0
		inst_feats = []
		inst_len_feats, inst_ang_feats = [], []
		for i in range(img_num):
			
			inst_coord = inst_coords[i]
			num_gts += inst_coord.size(0)
			#if num_gts >= max_person_num: break

			inst_feat = x[i, :, inst_coord[:, 0], inst_coord[:, 1]].t()
			inst_feats.append(inst_feat)

			inst_len_feat = len_feat[i, :, inst_coord[:, 0], inst_coord[:, 1]].t()
			inst_len_feats.append(inst_len_feat)
			inst_ang_feat = ang_feat[i, :, inst_coord[:, 0], inst_coord[:, 1]].t()
			inst_ang_feats.append(inst_ang_feat)
			#pred_lens.append(pred_len)

		inst_len_feats = torch.cat(inst_len_feats, dim=0)
		inst_ang_feats = torch.cat(inst_ang_feats, dim=0)

		pred_lens = self.len_head(inst_len_feats)
		pred_angs = self.ang_head(inst_ang_feats)
		#print(x.size(), inst_len_feats.size(), inst_ang_feats.size(), pred_lens.size(), pred_angs.size())

		pred_angs = pred_angs.reshape(-1, self.num_keypoints, self.angle_bins)
		gt_lens, gt_angs = gt_len_angles.chunk(2, dim=1)
		pos_mask = (gt_lens >=0.9).float()
		joint_num = pos_mask.sum(dim=1).clamp(min=1e-5)

		#print('pred_angs', pred_angs.size(), gt_angs.size(), gt_len_angles.size())
		#print('pred_lens', pred_lens.size(), gt_lens.size())

		l_total = torch.stack([pred_lens.exp(), gt_lens], -1)
		l_max = l_total.max(dim=2)[0]
		l_min = l_total.min(dim=2)[0]
		loss_len = (l_max / l_min).log()*pos_mask
		loss_len = loss_len.sum(dim=1)/joint_num
		loss_len = loss_len.mean()#loss_len.mean(dim=1).mean(dim=0)

		gt_angs = gt_angs.long().unsqueeze(dim=2)
		gt_angs_onehot = torch.zeros((gt_angs.size(0), gt_angs.size(1), self.angle_bins), device=gt_angs.device)
		gt_angs_onehot.scatter_(2, gt_angs, 1)
		#print(target.size(), pred_angles_pos.size(), target.sum())

		pred_angs = F.log_softmax(pred_angs, dim=2)
		loss_ang = -(pred_angs*gt_angs_onehot).sum(dim=2)*pos_mask
		loss_ang = loss_ang.sum(dim=1)/joint_num
		#loss_ang = loss_ang.mean(dim=1).mean(dim=0)
		loss_ang = loss_ang.mean()
		#print('loss_len', loss_len.size(), loss_ang.size())

		loss_cls = self.loss_cls(cls_scores, heatmaps, masks)

		#print(cls_scores.size(), heatmaps.size(), masks.size())
		#print()
		return loss_cls, loss_ang, loss_len

	def forward_test(self, x, img_metas):

		cls_scores = self.cls_head(x)[:, self.num_keypoints:, :, :, ].sigmoid()
		#prototypes = self.prototype(x)
		#coef_feats = self.coef_head(x)

		cls_scores_max = self.adaptive_pool(cls_scores)
		cls_scores_max = torch.eq(cls_scores_max, cls_scores).float()
		cls_scores = cls_scores * cls_scores_max

		results_list = []
		for i in range(cls_scores.size(0)):

			cls_score = cls_scores[i]
			#prototype = prototypes[i]
			#coef_pred = coef_feats[i]
			result = self.get_result_single(cls_score, x[i], img_metas[i])
			results_list.append(result)
			#draw_heatmap.append([result[1], result[2]])

		return results_list

	def get_result_single(self, cls_score, img_fea, img_meta):

		#print(cls_score.size(), prototype.size(), coef_feat.size())		
		h, w = cls_score.size()[1:]
		cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_cpoints)#.max(dim=1)
		img_fea = img_fea.permute(1, 2, 0).reshape(-1, self.in_channels)

		pred_scores, inst_feats = [], []
		all_topk_inds = []
		for i in range(self.num_cpoints):

			scores = cls_score[:, i]
			scores, topk_inds = scores.topk(self.nms_pre, dim=0)
			valid_inds = (scores > self.score_thr).nonzero().view(-1)

			if len(valid_inds) == 0: continue

			scores = scores[valid_inds]
			topk_inds = topk_inds[valid_inds]
			inst_feat = img_fea[topk_inds]

			pred_scores.append(scores)
			inst_feats.append(inst_feat)
			all_topk_inds.append(topk_inds)

		if len(pred_scores) == 0:
			pose_results = torch.zeros((0, self.num_keypoints, 3), device=cls_score.device)
			scores = torch.zeros((0), device=cls_score.device)
			return pose_results, scores

		pred_scores = torch.cat(pred_scores, dim=0)
		inst_feats = torch.cat(inst_feats, dim=0)
		all_topk_inds = torch.cat(all_topk_inds, dim=0).unsqueeze(dim=1)

		c_x, c_y = all_topk_inds%w, all_topk_inds//w

		#print('inst_feats', inst_feats.size())

		pred_lens = self.pose_length(inst_feats).exp()
		pred_angs = self.pose_angle(inst_feats)
		pred_angs = pred_angs.reshape(-1, self.num_keypoints, self.angle_bins)
		#print('pred_angs', pred_angs.size(), pred_lens.size())

		pred_angs = F.softmax(pred_angs, dim=2)
		pred_angs_prob, pred_angs_idx = torch.max(pred_angs, dim=2)
		pred_angs_idx = pred_angs_idx.float()
		pred_angs_idx = (pred_angs_idx*360.0/self.angle_bins)/180*math.pi
		#print('pred_lens', c_x.size(), pred_lens.size(), pred_angs_idx.size())
		x = c_x + pred_lens * torch.cos(pred_angs_idx)
		y = c_y + pred_lens * torch.sin(pred_angs_idx)
		visible = pred_angs_prob
		#print('x', x.size(), y.size(), visible.size())

		pose_results = torch.cat([x.unsqueeze(dim=2)*4, y.unsqueeze(dim=2)*4, visible.unsqueeze(dim=2)], dim=2)
		pose_results[:, :, 0:2] = pose_results[:, :, 0:2]/img_meta['scale_factor']
		#print('scale_factor', img_meta['scale_factor'])
		keep, _ = oks_nms(pose_results.cpu().numpy(), pred_scores.cpu().numpy(), self.nms_thr, self.sigmas)
		pose_results, pred_scores = pose_results[keep], pred_scores[keep]
		#print('pose_results', pose_results.size(), pose_results.size(), visible.size())
		#print('pred_lens2', pred_lens[keep])
		return pose_results, pred_scores

	def forward_test_flip(self, x, img_metas):

		#print('x', x.size())
		cls_feat = x
		len_feat = x
		ang_feat = x

		for cls_layer in self.cls_convs:
			cls_feat = cls_layer(cls_feat)
		cls_scores = self.cls_head(cls_feat).sigmoid()

		for len_layer in self.len_convs:
			len_feat = len_layer(len_feat)

		for ang_layer in self.ang_convs:
			ang_feat = ang_layer(ang_feat)


		heatmaps = cls_scores[:, :self.num_keypoints, :, :, ]
		heatmaps, heatmaps_flip = torch.chunk(heatmaps, 2, dim=0)
		heatmaps_flip = heatmaps_flip.flip(dims=[3])
		heatmaps_flip = heatmaps_flip[:, self.flip_index, :, :]
		heatmaps = (heatmaps+heatmaps_flip)/2.0



		cls_scores = cls_scores[:, self.num_keypoints:, :, :, ]
		cls_scores, cls_scores_flip = torch.chunk(cls_scores, 2, dim=0)
		cls_scores_flip = cls_scores_flip.flip(dims=[3])
		cls_scores = (cls_scores+cls_scores_flip)/2.0

		cls_scores_max = self.adaptive_pool(cls_scores)
		cls_scores_max = torch.eq(cls_scores_max, cls_scores).float()
		cls_scores = cls_scores * cls_scores_max


		len_feat, len_feat_flip = torch.chunk(len_feat, 2, dim=0)
		len_feat_flip = len_feat_flip.flip(dims=[3])

		ang_feat, ang_feat_flip = torch.chunk(ang_feat, 2, dim=0)
		ang_feat_flip = ang_feat_flip.flip(dims=[3])



		results_list = []
		for i in range(cls_scores.size(0)):

			result = self.get_result_single_flip(
				cls_scores[i], 
				len_feat[i], len_feat_flip[i],
				ang_feat[i], ang_feat_flip[i], img_metas[i], heatmaps[i])
			results_list.append(result)

		return results_list

	def get_result_single_flip(self, cls_score, len_feat, len_feat_flip, ang_feat, ang_feat_flip,img_meta, heatmap): 

		h, w = cls_score.size()[1:]
		cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_cpoints)#.max(dim=1)

		len_feat = len_feat.permute(1, 2, 0).reshape(-1, self.feat_channels)
		ang_feat = ang_feat.permute(1, 2, 0).reshape(-1, self.feat_channels)
		len_feat_flip = len_feat_flip.permute(1, 2, 0).reshape(-1, self.feat_channels)
		ang_feat_flip = ang_feat_flip.permute(1, 2, 0).reshape(-1, self.feat_channels)

		pred_scores = []
		inst_len_feats, inst_len_feats_flip = [], []
		inst_ang_feats, inst_ang_feats_flip = [], []
		all_topk_inds = []
		cpoint_inds = []
		for i in range(self.num_cpoints):
			#if i != 1: continue
			scores = cls_score[:, i]
			scores, topk_inds = scores.topk(self.nms_pre, dim=0)
			valid_inds = (scores > self.score_thr).nonzero().view(-1)

			if len(valid_inds) == 0: continue

			scores = scores[valid_inds]
			topk_inds = topk_inds[valid_inds]

			inst_len_feat = len_feat[topk_inds]
			inst_len_feat_flip = len_feat_flip[topk_inds]

			inst_ang_feat = ang_feat[topk_inds]
			inst_ang_feat_flip = ang_feat_flip[topk_inds]

			pred_scores.append(scores)
			inst_len_feats.append(inst_len_feat)
			inst_len_feats_flip.append(inst_len_feat_flip)
			inst_ang_feats.append(inst_ang_feat)
			inst_ang_feats_flip.append(inst_ang_feat_flip)
			cpoint_inds.append(i*torch.ones(scores.size(0), device=cls_score.device))

			all_topk_inds.append(topk_inds)

		if len(pred_scores) == 0:
			pose_results = torch.zeros((0, self.num_keypoints, 3), device=cls_score.device)
			scores = torch.zeros((0), device=cls_score.device)
			return pose_results, scores


		cpoint_inds = torch.cat(cpoint_inds, dim=0).long()
		pred_scores = torch.cat(pred_scores, dim=0)
		inst_len_feats = torch.cat(inst_len_feats, dim=0)
		inst_len_feats_flip = torch.cat(inst_len_feats_flip, dim=0)
		inst_ang_feats = torch.cat(inst_ang_feats, dim=0)
		inst_ang_feats_flip = torch.cat(inst_ang_feats_flip, dim=0)
		all_topk_inds = torch.cat(all_topk_inds, dim=0).unsqueeze(dim=1)

		c_x, c_y = all_topk_inds%w, all_topk_inds//w
		#c_x, c_y  = c_x+0.5, c_y +0.5
		#print('inst_feats', inst_feats.size())

		pred_lens = self.len_head(inst_len_feats).exp()
		pred_angs = self.ang_head(inst_ang_feats)
		pred_angs = pred_angs.reshape(-1, self.num_keypoints, self.angle_bins)

		pred_angs = F.softmax(pred_angs, dim=2)
		pred_angs_prob, pred_angs_idx = torch.max(pred_angs, dim=2)
		pred_angs_idx = pred_angs_idx.float()
		pred_angs_idx = (pred_angs_idx*360.0/self.angle_bins)/180*math.pi

		x = c_x + pred_lens * torch.cos(pred_angs_idx)
		y = c_y + pred_lens * torch.sin(pred_angs_idx)
		visible = pred_angs_prob
		pose_results = torch.cat([x.unsqueeze(dim=2)*4, y.unsqueeze(dim=2)*4, visible.unsqueeze(dim=2)], dim=2)



		pred_lens_flip = self.len_head(inst_len_feats_flip).exp()
		pred_angs_flip = self.ang_head(inst_ang_feats_flip)
		pred_angs_flip = pred_angs_flip.reshape(-1, self.num_keypoints, self.angle_bins)

		pred_angs_flip = F.softmax(pred_angs_flip, dim=2)
		pred_angs_prob_flip, pred_angs_idx_flip = torch.max(pred_angs_flip, dim=2)
		pred_angs_idx_flip = pred_angs_idx_flip.float()
		pred_angs_idx_flip = (pred_angs_idx_flip*360.0/self.angle_bins)/180*math.pi

		x_flip = c_x - pred_lens_flip * torch.cos(pred_angs_idx_flip)+1
		y_flip = c_y + pred_lens_flip * torch.sin(pred_angs_idx_flip)
		visible_flip = pred_angs_prob_flip
		pose_results_flip = torch.cat([x_flip.unsqueeze(dim=2)*4, y_flip.unsqueeze(dim=2)*4, visible_flip.unsqueeze(dim=2)], dim=2)


		#print('scale_factor', img_meta['scale_factor'])
		pose_results = (pose_results + pose_results_flip[:, self.flip_index, :])/2
		#pose_results = pose_results_flip[:, self.flip_index, :]
		pose_results[:, :, 0:2] = pose_results[:, :, 0:2]/img_meta['scale_factor']


		pose_results, pred_scores = self.adjust_output(pose_results[:,:,0:2], pred_scores, heatmap.unsqueeze(dim=0), 4/img_meta['scale_factor'])


		keep, keep_ind = oks_nms(pose_results.cpu().numpy(), pred_scores.cpu().numpy(), self.nms_thr, self.sigmas)

		#pose_results = self.refine_result(pose_results, keep, keep_ind, cpoint_inds)
		#pred_scores = pred_scores[keep]
		#pose_results, pred_scores = pose_results[keep], pred_scores[keep]
		pose_results, pred_scores = self.pose_fusion(pose_results, keep, keep_ind)





		#keep, _ = oks_nms(pose_results.cpu().numpy(), pred_scores.cpu().numpy(), self.nms_thr, self.sigmas)
		#pose_results, pred_scores = pose_results[keep], pred_scores[keep]
		
		return pose_results, pred_scores


	def pose_fusion(self, poses, keep, nms_ind):
		num_keypoints = poses.size(1)
		fused_poses = torch.zeros((len(keep), num_keypoints, 3)).to(poses.device)

		for idx, i in enumerate(keep):
			fused_poses[idx] = poses[i]
			for j in nms_ind[idx]:
				nms_pose = poses[j]
				for k in range(num_keypoints):
					if nms_pose[k, 2] > fused_poses[idx, k, 2]:
						fused_poses[idx, k, :] = nms_pose[k]

		fused_scores = fused_poses[:, :, 2].mean(dim=1)
		return fused_poses, fused_scores


	def adjust_output(self, poses, scores, heatmap, scale = 1.0):

		h, w = heatmap.size()[2:]
		#print(h*scale, w*scale)
		heatmap= torch.nn.functional.interpolate(heatmap,size=(int(h*scale), int(w*scale)), align_corners=True, mode='bilinear')
		#print('heatmap', heatmap.size(), poses.size())
		heatmap = heatmap[0]
		num_people, num_keypoints = poses.size()[:2]
		heatval = np.zeros((num_people, num_keypoints, 1))

		for i in range(num_people):
			for j in range(num_keypoints):
				k1, k2 = int(poses[i, j, 0]), int(poses[i, j, 0]) + 1
				k3, k4 = int(poses[i, j, 1]), int(poses[i, j, 1]) + 1
				u = poses[i, j, 0] - int(poses[i, j, 0])
				v = poses[i, j, 1] - int(poses[i, j, 1])
				if k2 < heatmap.shape[2] and k1 >= 0 and k4 < heatmap.shape[1] and k3 >= 0:
					heatval[i, j, 0] = heatmap[j, k3, k1] * (1 - v) * (1 - u) + heatmap[j, k4, k1] * (1 - u) * v + \
						heatmap[j, k3, k2] * u * (1 - v) + heatmap[j, k4, k2] * u * v

		scores = torch.tensor(
			scores[:, None].expand(-1, num_keypoints)[:, :, None].cpu().numpy() * heatval).float()
		poses = torch.cat([poses.cpu(), scores.cpu()], dim=2)
		poses = poses.cpu().numpy()

		scores = np.mean(poses[:, :, 2], axis=1)

		return torch.tensor(poses).to(heatmap.device), torch.tensor(scores).to(heatmap.device)


	def refine_result(self, pose_results, keep, keep_ind, cpoint_inds):

		cpoints_idx = [[0, 1, 2, 3, 4], [5,6, 11,12], [11, 12, 13, 14, 15, 16]]

		cpoint_weight = [[], [], []]
		for i in range(17):
			for j in range(3):
				if i in cpoints_idx[j]:
					cpoint_weight[j].append(1)
				elif j == 1:
					cpoint_weight[j].append(0.01)
				else:
					cpoint_weight[j].append(0.0001)
		cpoint_weight = torch.tensor(cpoint_weight, device=pose_results.device)
		#print('cpoint_weight', cpoint_weight.size())
		refined_poses = []
		for i in range(len(keep)):

			if len(keep_ind[i]) == 0:
				refined_poses.append(pose_results[keep[i]])
				continue

			group_inds = keep_ind[i]
			group_inds.insert(0, keep[i])
			#print('group_inds', i, group_inds)

			cpoint_ind = cpoint_inds[group_inds]
			refined_pose = pose_results[group_inds]

			#for j in range(3):
			#print(i, 'cpoint_ind', cpoint_ind.size(), cpoint_ind)
			fuse_weight = cpoint_weight[cpoint_ind]
			fuse_weight = fuse_weight/fuse_weight.sum(dim=0, keepdim=True)
			#print(i, cpoint_ind.size(), refined_pose.size(), fuse_weight.size())

			refined_pose = refined_pose*fuse_weight.unsqueeze(dim=2)
			refined_pose = refined_pose.sum(dim=0)
			#refined_pose = refined_pose.mean(dim=0)
			refined_poses.append(refined_pose)

		refined_poses = torch.stack(refined_poses, dim=0)
		#print('refined_poses', refined_poses.size())
		return refined_poses


	def adaptive_pool(self, heatmap):

		size_threshold1 = 300
		size_threshold2 = 200
		#print('adaptive_pool', heatmap.size())
		map_size = (heatmap.shape[3] + heatmap.shape[2]) / 2.0
		if map_size > size_threshold1:
			heatmap = F.max_pool2d(heatmap, 7, 1, 3)
		elif map_size > size_threshold2:
			heatmap = F.max_pool2d(heatmap, 5, 1, 2)
		else:
			heatmap = F.max_pool2d(heatmap, 3, 1, 1)
		#print('adaptive_pool', heatmap.size())
		return heatmap





	# def up_interpolate(self, x, size, mode='bilinear', aligh_corners=True):
	# 	H=x.size()[2]
	# 	W=x.size()[3]
	# 	scale_h=int(size[0]/H)
	# 	scale_w=int(size[1]/W)
	# 	inter_x= torch.nn.functional.interpolate(x,size=[size[0]-scale_h+1,size[1]-scale_w+1],align_corners=aligh_corners,mode=mode)
	# 	padd= torch.nn.ReplicationPad2d((0,scale_w-1,0,scale_h-1))
	# 	return padd(inter_x)

