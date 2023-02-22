import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

	def __init__(self,
				 use_sigmoid=True,
				 gamma=2.0,
				 alpha=0.25,
				 reduction='mean',
				 loss_weight=1.0):
		super(FocalLoss, self).__init__()
		assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
		self.use_sigmoid = use_sigmoid
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction
		self.loss_weight = loss_weight
		#print('focal loss init', gamma, alpha, use_sigmoid, reduction, loss_weight)

	def forward(self,
				pred,
				target,
				weight=None,
				avg_factor=None):

		alpha = self.alpha
		gamma = self.gamma
		pred_sigmoid = pred.sigmoid()
		target = target.type_as(pred)
		pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
		focal_weight = (alpha * target + (1 - alpha)*(1 - target)) * pt.pow(gamma)
		#print('focal loss', avg_factor, pred.size(), target.size(), self.reduction, weight)
		loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
		#loss = loss.sum() / avg_factor
		return loss.sum() / avg_factor


class HeatmapLoss(nn.Module):
		def __init__(self):
				super().__init__()

		def forward(self, pred, gt, mask):
				assert pred.size() == gt.size()
				loss = ((pred - gt) ** 2) * mask
				loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
				return loss


class Neg_loss(nn.Module):
	def __init__(self, alpha=2, beta=4):
		super(Neg_loss, self).__init__()
		self.alpha = alpha
		self.beta = beta

	def forward(self, pred, gt, mask=None):
		''' Modified focal loss. Exactly the same as CornerNet.
			Runs faster and costs a little bit more memory
			Arguments:
			  pred (batch x c x h x w)
			  gt_regr (batch x c x h x w)
		'''
		pos_inds = gt.eq(1).float()
		neg_inds = gt.lt(1).float()
		pred = torch.clamp(pred.sigmoid_(), min=1e-4, max=1-1e-4)

		if mask is not None:
			pos_inds = pos_inds * mask
			neg_inds = neg_inds * mask

		#print(pred.size(), pos_inds.size(), mask.size(), gt.size(), )

		neg_weights = torch.pow(1 - gt, self.beta)

		loss = 0

		pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
		neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

		num_pos = pos_inds.float().sum()
		pos_loss = pos_loss.sum()
		neg_loss = neg_loss.sum()

		if num_pos == 0:
			loss = loss - neg_loss
		else:
			loss = loss - (pos_loss + neg_loss) / num_pos
		return loss