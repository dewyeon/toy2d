import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA as Fr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from scipy.stats import special_ortho_group
from attn_modules.res_invertAttn_step import resInvertISDP
from attn_modules.modules import InvertibleConv1x1, ParallelPermute
import wandb
from einops import rearrange

class ResInvAttnBlock(Fm.AllInOneBlock):   
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
    
        super(ResInvAttnBlock, self).__init__(dims_in, dims_c, subnet_constructor)
        
        self.cfg = cfg
        self.ActNorm = Fm.ActNorm(dims_in, dims_c)
        self.inv1x1conv = InvertibleConv1x1(dims_in[0][0], LU_decomposed=True)
        self.seed = self.cfg.seed# csflow 에서는 같은 coupling block 당 하나씩 동일시드
        if self.cfg.block_ver == 'cslayer2-5':
            self.csflow_perm = [ParallelPermute(dims_in, self.seed + i) for i in range(self.cfg.L_layers)]
        else:
            self.csflow_perm = ParallelPermute(dims_in, self.seed) 
        self.cfg.seed += self.cfg.L_layers
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels + self.cfg.L_layers, 2 * self.splits[1])
        self.subnet2 = subnet_constructor(self.splits[0] + self.condition_channels + self.cfg.L_layers, 2 * self.splits[1])

        if cfg.A == 'mat':
            self.resInvertISDP1_1 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            self.resInvertISDP1_2 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 1:
                self.resInvertISDP2_1 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.resInvertISDP2_2 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 2:
                self.resInvertISDP3_1 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.resInvertISDP3_2 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 3:
                self.resInvertISDP4_1 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.resInvertISDP4_2 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)


        if self.cfg != None and self.cfg.use_wandb:
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_table)


    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''

        if self.cfg.block_ver == 'cs2-5': # 2-5 + csflow permutation
            """
            0.CS-Flow Permutation
            1.Attention
            2.Affine
            3.Attention
            4.Affine
            """
            log_jac_det_list = [0 for l in range(self.cfg.L_layers)]
            x_in = [0 for l in range(self.cfg.L_layers)]
            ### csflow permutation
            for l in range(self.cfg.L_layers):
                x_in[l] = self.csflow_perm(x[0][l], rev=rev)
        
            x, log_jac_det_list = self.resInvertISDP1_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
    
            
            x = torch.cat(x, dim=0)
            log_jac_det = torch.cat(log_jac_det_list, dim=0)
            cond_vec = torch.cat(c[0], dim=0)
    
            x1, x2 = torch.split(x, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, cond_vec], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2

            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = [[] for l in range(self.cfg.L_layers)]
            split_res = x_out.shape[0] // self.cfg.L_layers
            for l in range(self.cfg.L_layers):
                out_list[l] = x_out[l*split_res:(l+1)*split_res]
                log_jac_det_list[l] = log_jac_det[l*split_res:(l+1)*split_res]
            
            x_out_list, log_jac_det_list = self.resInvertISDP1_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = torch.cat(x_out_list, dim=0)
            log_jac_det = torch.cat(log_jac_det_list, dim=0)
            
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, cond_vec], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2

            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            out_list = [[] for l in range(self.cfg.L_layers)]
            for l in range(self.cfg.L_layers):
                out_list[l] = x_out[l*split_res:(l+1)*split_res]
            
        return (out_list,), log_jac_det

# Flow level : [squeeze - actnorm - invertible 1x1 - affine coupling - invertible attention - split - squezze]