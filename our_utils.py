import numpy as np
import torch
import torch.nn.functional as F
import einops
from einops import rearrange
import math 

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))

def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)

def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m)) #[768] -> flat to [768,1]    
    return torch.cat(flat_maps, dim=1)[..., None]

class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score()
        
        return save_weights

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_logp(C, z, logdet_J):
    # import pdb; pdb.set_trace()
    
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    # logp = - C * 0.5 * math.log(math.pi * 2) - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp
    
def get_logp_z(z):
    # import pdb; pdb.set_trace()
    C = 2
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1)
    # logp = - C * 0.5 * math.log(math.pi * 2) - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

def get_csflow_loss_token_channel_1(z, jac, B, H, W):
    z_sum_channel = torch.sum(z, dim=-1) # [HW]
    z_sum_channel = z_sum_channel**2 # [HW]

    loss = torch.mean(0.5 * z_sum_channel - jac) / z.shape[1]
    return loss

def get_csflow_loss_token_channel_2(z, jac, B, H, W):
    z_sum_channel = torch.sum(z, dim=-1) # [HW]
    z_sum_channel = z_sum_channel**2 # [HW]

    loss = torch.mean(0.5 * z_sum_channel - jac)
    return loss

def get_csflow_loss_token_channel_3(z, jac, B, H, W, C):
    z_sum_channel = torch.sum(z, dim=-1) # [HW]
    z_sum_channel = z_sum_channel**2 # [HW]

    loss = C * _GCONST_ - torch.mean(0.5 * z_sum_channel - jac) / z.shape[1]
    return loss

def get_csflow_loss_token_channel_4(z, jac, B, H, W, C):
    z_sum_channel = torch.sum(z, dim=-1) # [HW]
    z_sum_channel = z_sum_channel**2 # [HW]

    loss = C * _GCONST_ - torch.mean(0.5 * z_sum_channel - jac)
    return loss

def get_csflow_loss_token_channel_5(z, jac, B, H, W):
    z_sum_channel = torch.sum(z, dim=-1) # [HW]
    z_sum_channel = z_sum_channel**2 # [HW]

    loss = H*W * _GCONST_ - torch.mean(0.5 * z_sum_channel - jac) / z.shape[1]
    return loss

def get_csflow_loss_token_channel_6(z, jac, B, H, W):
    z_sum_channel = torch.sum(z, dim=-1) # [HW]
    z_sum_channel = z_sum_channel**2 # [HW]

    loss = H*W * _GCONST_ - torch.mean(0.5 * z_sum_channel - jac)
    return loss



def get_csflow_loss(z, jac, B, H, W):
    # z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
    # jac = sum(jac)

    z = rearrange(z, '(B H W L) C -> B (H W L C)', B=B, H=H, W=W)
    jac = rearrange(jac, '(B H W L) -> B (H W L)', B=B, H=H, W=W)

    jac = jac.sum(dim=1)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

def get_csflow_loss2(z, jac, B, H, W):
    # z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
    # jac = sum(jac)

    z = rearrange(z, '(B H W) C -> B (H W C)', B=B, H=H, W=W)
    jac = rearrange(jac, '(B H W) -> B (H W)', B=B, H=H, W=W)
    
    jac = jac.sum(dim=1)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]
def get_csflow_loss3(z, jac, B, H, W):
    z = rearrange(z, '(B H W) C -> B (H W C)', B=B, H=H, W=W)
    jac = rearrange(jac, '(B H W) -> B (H W)', B=B, H=H, W=W)
    
    jac = jac.sum(dim=1)
    return (0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def get_csflow_loss_token(z, jac, B, H, W):
    # z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
    # jac = sum(jac)
    z = rearrange(z, '(B H W) C -> B (H W C)', B=B, H=H, W=W)
    jac = rearrange(jac, '(B H W) -> B (H W)', B=B, H=H, W=W)
    jac = jac.sum(dim=1)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

def get_loss_token(HW, z, logdet_J):
    # import pdb; pdb.set_trace()
    loss = (0.5 * torch.sum(z ** 2, 1) - logdet_J) / HW
    return loss

def get_loss_token_dim(HW, C, z, logdet_J):
    # import pdb; pdb.set_trace()
    loss = ((0.5 * torch.sum(z ** 2, 1) - logdet_J) * HW) / C
    return loss


def get_loss(z, jac): # differNet
    # z : [BHW, C]
    # import pdb; pdb.set_trace()
    
    # get token
    # jac = sum(jac)
    # return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]
    return (0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
