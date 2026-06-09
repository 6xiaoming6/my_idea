import torch
import torch.nn.functional as F

def _denom(mask, c, eps=1e-6):
    return mask.sum() * c + eps

def masked_mae(pred, target, obs_mask):
    miss = 1.0 - obs_mask
    return (torch.abs(pred-target)*miss).sum() / _denom(miss, pred.shape[1])

def masked_rmse(pred, target, obs_mask):
    miss = 1.0 - obs_mask
    return torch.sqrt((((pred-target)**2)*miss).sum() / _denom(miss, pred.shape[1]))

def masked_mape(pred, target, obs_mask, eps=1e-3):
    miss = 1.0 - obs_mask
    return (torch.abs((pred-target)/(torch.abs(target)+eps))*miss).sum() / _denom(miss, pred.shape[1])

def masked_smooth_l1_loss(pred, target, obs_mask):
    miss = 1.0 - obs_mask
    loss = F.smooth_l1_loss(pred*miss, target*miss, reduction='sum')
    return loss / _denom(miss, pred.shape[1])

@torch.no_grad()
def compute_metrics(pred, target, obs_mask):
    return {"mae": float(masked_mae(pred,target,obs_mask).cpu()),
            "rmse": float(masked_rmse(pred,target,obs_mask).cpu()),
            "mape": float(masked_mape(pred,target,obs_mask).cpu())}
