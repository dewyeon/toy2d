import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA as Fr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from scipy.stats import special_ortho_group
from attn_modules.magicAttn_step import MagicISDP
import wandb

class MagicAttnBlock(Fm.AllInOneBlock):   
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 affine_clamping: float = 2.,
                 gin_block: bool = False,
                 global_affine_init: float = 1.,
                 global_affine_type: str = 'SOFTPLUS',
                 permute_soft: bool = False,
                 learned_householder_permutation: int = 0,
                 reverse_permutation: bool = False,
                 cfg = None,
                 split_dim=1,
                 n_head=1, is_multiHead=False):
    
        super(MagicAttnBlock, self).__init__(dims_in, dims_c, subnet_constructor)
        
        self.cfg = cfg
        self.MagicISDP = MagicISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
        
        if self.cfg != None and self.cfg.use_wandb:
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_table)


    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        #import pdb; pdb.set_trace()
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1
            
        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)


        """
        After Coupling block, applying MagicISDP
        """
        if not rev:
            x_out, log_jac_det = self.MagicISDP(x_out, logdet=log_jac_det, reverse=False, permute=False)
        else:
            x_out, log_jac_det = self.MagicISDP(x_out, logdet=log_jac_det, reverse=True, permute=False)
            

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        
        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac


        if self.cfg != None and self.cfg.use_wandb:
            wandb.log({'global_scale_logdet': torch.mean(global_scaling_jac), 'FlowStep_logdet': torch.mean(log_jac_det)})
    

        return (x_out,), log_jac_det

# Flow level : [squeeze - actnorm - invertible 1x1 - affine coupling - invertible attention - split - squezze]