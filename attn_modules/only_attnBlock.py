import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA as Fr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from scipy.stats import special_ortho_group
from attn_modules.attention_step import ISDP

class Only_isdpAttnBlock(Fm.AllInOneBlock):   
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 affine_clamping: float = 2.,
                 gin_block: bool = False,
                 global_affine_init: float = 1.,
                 global_affine_type: str = 'SOFTPLUS',
                 permute_soft: bool = False,
                 learned_householder_permutation: int = 0,
                 reverse_permutation: bool = False,
                 cfg = None):
    
        super(Only_isdpAttnBlock, self).__init__(dims_in, dims_c, subnet_constructor)
        
        self.config = cfg
        self.ISDP = ISDP(dims_in[0][0])
        


    def forward(self, x, c=[], rev=False, jac=True):
        # '''See base class docstring'''
        # if self.householder:
        #     self.w_perm = self._construct_householder_permutation()
        #     if rev or self.reverse_pre_permute:
        #         self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        # elif self.reverse_pre_permute:
        #     x = (self._pre_permute(x[0], rev=False),)

        # x1, x2 = torch.split(x[0], self.splits, dim=1)

        # if self.conditional:
        #     x1c = torch.cat([x1, *c], 1)
        # else:
        #     x1c = x1

        # if not rev:
        #     a1 = self.subnet(x1c)
        #     x2, j2 = self._affine(x2, a1)
        # else:
        #     a1 = self.subnet(x1c)
        #     x2, j2 = self._affine(x2, a1, rev=True)

        # log_jac_det = j2
        # x_out = torch.cat((x1, x2), 1)
        # log_jac_det = torch.zeros(1, 768, 196).to(torch.device("cuda"))
        log_jac_det = torch.zeros(16*16).to(torch.device("cuda"))
        # input : (1,3,4,4)
        
        # import pdb; pdb.set_trace()

        """
        After Coupling block, applying ISDP
        """
        if not rev:
            x_out, log_jac_det = self.ISDP(x[0], logdet=log_jac_det, reverse=False, permute=False)
        else:
            x_out, log_jac_det = self.ISDP(x[0], logdet=log_jac_det, reverse=True, permute=False)
            
        # import pdb; pdb.set_trace()

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        # elif self.reverse_pre_permute:
        #     x_out = self._pre_permute(x_out, rev=True)

            
        # # add the global scaling Jacobian to the total.
        # # trick to get the total number of non-channel dimensions:
        # # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

# Flow level : [squeeze - actnorm - invertible 1x1 - affine coupling - invertible attention - split - squezze]