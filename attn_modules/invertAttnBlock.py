import FrEIA.framework as Ff
import FrEIA.modules as Fm
import FrEIA as Fr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from scipy.stats import special_ortho_group
from attn_modules.invertAttn_step import InvertISDP
from attn_modules.invertAttn_step_vec import InvertISDPvec
from attn_modules.modules import InvertibleConv1x1, ParallelPermute
import wandb
from einops import rearrange

class InvAttnBlock(Fm.AllInOneBlock):   
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
    
        super(InvAttnBlock, self).__init__(dims_in, dims_c, subnet_constructor)
        
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

        if cfg.A == 'vec':
            self.InvertISDP1_1 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            self.InvertISDP1_2 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 1:
                self.InvertISDP2_1 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.InvertISDP2_2 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 2:
                self.InvertISDP3_1 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.InvertISDP3_2 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 3:
                self.InvertISDP4_1 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.InvertISDP4_2 = InvertISDPvec(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)      

        elif cfg.A == 'mat':
            self.InvertISDP1_1 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            self.InvertISDP1_2 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 1:
                self.InvertISDP2_1 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.InvertISDP2_2 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 2:
                self.InvertISDP3_1 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.InvertISDP3_2 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
            if self.cfg.attn_step > 3:
                self.InvertISDP4_1 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
                self.InvertISDP4_2 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            
        if self.cfg != None and self.cfg.use_wandb:
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_table)


    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        # if self.householder:
        #     self.w_perm = self._construct_householder_permutation()
        #     if rev or self.reverse_pre_permute:
        #         self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        # if rev:
        #     x, global_scaling_jac = self._permute(x[0], rev=True)
        #     x = (x,)
        # elif self.reverse_pre_permute:
        #     x = (self._pre_permute(x[0], rev=False),)

        if self.cfg.block_ver == '0':
            x_out = x[0]; log_jac_det=0
            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)
        elif self.cfg.block_ver == '2-1':
            
            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det = j2 # [BHW]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

        elif self.cfg.block_ver == '2-2':
            '''affine -> inv_attn1'''
            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det = j2 # [BHW]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
        
            '''affine -> inv_attn2'''
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

        elif self.cfg.block_ver == '2-3-1':
            '''inv_attn'''
            x_out = x[0]; log_jac_det=0
            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            # x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)

            '''affine'''
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW] 
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            
            '''inv_attn'''
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

        elif self.cfg.block_ver == '2-3-2':
            '''inv_attn'''
            x_out = x[0]; log_jac_det=0
            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)

            '''affine전에 x1,x2 cross'''
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            
            '''inv_attn'''
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

        elif self.cfg.block_ver == '2-4':
            x_out = x[0]; log_jac_det=0
            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]


        elif self.cfg.block_ver == '2-5':
            '''inv_attn -> affine -> inv_attn -> affine'''
            x_out = x[0]; log_jac_det=0
            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)

            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW] 
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            # x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW] 
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

        elif self.cfg.block_ver == '2-6': # 2-5 + permutation
            """
            1.Attention
            2.Affine
            3.Attention
            4.Affine
            5.Permutation
            """
            log_jac_det=0
            x_out, log_jac_det = self.InvertISDP1_1(x[0], cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            # permutation
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
            n_pixels = x_out[0, :1].numel()
            log_jac_det += (-1)**rev * n_pixels * global_scaling_jac
        
        elif self.cfg.block_ver == 'freia2-5': # 2-5 + freia permutation
            """
            0.Freia Permutation
            1.Attention
            2.Affine
            3.Attention
            4.Affine
            """
            log_jac_det=0
            ### freia permutation
            import pdb; pdb.set_trace()
            
            x_out, global_scaling_jac = self._permute(x[0], rev=False)
            n_pixels = x_out[0, :1].numel()
            log_jac_det += (-1)**rev * n_pixels * global_scaling_jac

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

        elif self.cfg.block_ver == 'cs2-5': # 2-5 + csflow permutation
            """
            0.CS-Flow Permutation
            1.Attention
            2.Affine
            3.Attention
            4.Affine
            """
            log_jac_det=0
            ### csflow permutation
            x_out = self.csflow_perm(x[0], rev=rev)

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
        elif self.cfg.block_ver == 'cslayer2-5': # 2-5 + csflow permutation for layer wise
            """
            0.CS-Flow Permutation
            1.Attention
            2.Affine
            3.Attention
            4.Affine
            """
            log_jac_det=0
            ### csflow permutation
            per_layer = x[0].shape[0] // self.cfg.L_layers
            x_input = [[] for l in range(self.cfg.L_layers)]
            for l in range(self.cfg.L_layers):
                x_token = x[0][l*per_layer:(l+1)*per_layer]
                x_input[l] = self.csflow_perm[l](x_token, rev=rev)
            x_out = torch.cat(x_input, dim=0)


            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

        elif self.cfg.block_ver == '1x12-5': # 2-5 + 1x1 convolution permutation
            """
            0.1x1 convolution
            1.Attention
            2.Affine
            3.Attention
            4.Affine
            """
            log_jac_det=0; x_out=x[0]
            ### 1x1 conv permutation
            p_size = 14  # temporary
            x_out = rearrange(x_out, '(b h w) c -> b c h w', h=p_size, w=p_size)
            x_out, log_jac_det = self.inv1x1conv(x_out, log_jac_det, reverse=rev)
            x_out = rearrange(x_out, 'b c h w -> (b h w) c')
            
            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            if not rev:
                a1 = self.subnet2(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet2(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)
            log_jac_det += j2 # [BHW]
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

        elif self.cfg.block_ver == '3-1': # X
            '''Actnorm -> invattn1 -> inv_attn2 -> affine'''
            # x.shape [BxHxW, C]
            if not rev:
                x_out, log_jac_det = self.ActNorm(x, rev=False)
            else:
                x_out, log_jac_det = self.ActNorm(x, rev=True)
            # x_out.shape 
            x_out = x_out[0]

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)


            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]

        elif self.cfg.block_ver == '3-2-1':
            '''Actnorm -> invattn1 -> affine -> inv_attn2 -> affine'''
            # x.shape [BxHxW, C]
            if not rev:
                x_out, log_jac_det = self.ActNorm(x, rev=False)
            else:
                x_out, log_jac_det = self.ActNorm(x, rev=True)
            # x_out.shape 
            x_out = x_out[0]

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]
            
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x2, x1), 1)
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x2, x1), 1)

        elif self.cfg.block_ver == '3-2-2':
            '''Actnorm -> invattn1 -> affine -> inv_attn2 -> affine''' # before affine, cross x1,x2
            # x.shape [BxHxW, C]
            if not rev:
                x_out, log_jac_det = self.ActNorm(x, rev=False)
            else:
                x_out, log_jac_det = self.ActNorm(x, rev=True)
            # x_out.shape 
            x_out = x_out[0]

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]
            
            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x1, x2), 1)
        
            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x[0], self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW]
            x_out = torch.cat((x2, x1), 1)

        elif self.cfg.block_ver == '3-3':
            '''actnorm -> inv_attn -> affine -> inv_attn -> affine'''
            # x.shape [BxHxW, C]
            if not rev:
                x_out, log_jac_det = self.ActNorm(x, rev=False)
            else:
                x_out, log_jac_det = self.ActNorm(x, rev=True)
            # x_out.shape 
            x_out = x_out[0]

            x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)

            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW] 
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
            # x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)

            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]

            if self.conditional:
                x1c = torch.cat([x1, *c], 1) # [BHW, C/2 + cond]
            else:
                x1c = x1
            
            if not rev:
                a1 = self.subnet(x1c) # [BHW, C]
                x2, j2 = self._affine(x2, a1)
            else:
                a1 = self.subnet(x1c)
                x2, j2 = self._affine(x2, a1, rev=True)

            log_jac_det += j2 # [BHW] 
            '''affine 후 x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

        elif self.cfg.block_ver == 'Base+our':
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


            if not rev:
                x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
                x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
            
                x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
                x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)
            else:
                return # not implemented yet


            if not rev:
                x_out, global_scaling_jac = self._permute(x_out, rev=False)
            elif self.reverse_pre_permute:
                x_out = self._pre_permute(x_out, rev=True)

        
            # add the global scaling Jacobian to the total.
            # trick to get the total number of non-channel dimensions:
            # number of elements of the first channel of the first batch member
            n_pixels = x_out[0, :1].numel()
            log_jac_det += (-1)**rev * n_pixels * global_scaling_jac


        """
        After Coupling block, applying InvertISDP
        """

        # if not rev:
             
        #     x_out, log_jac_det = self.InvertISDP1_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #     x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
        
        #     x_out, log_jac_det = self.InvertISDP1_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #     x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)
            
            
        #     if self.cfg.attn_step > 1:            
        #         x_out, log_jac_det = self.InvertISDP2_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #         x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
            
        #         x_out, log_jac_det = self.InvertISDP2_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #         x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)
        
        #     if self.cfg.attn_step > 2:            
        #         x_out, log_jac_det = self.InvertISDP3_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #         x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
            
        #         x_out, log_jac_det = self.InvertISDP3_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #         x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)
                
        #     if self.cfg.attn_step > 3:            
        #         x_out, log_jac_det = self.InvertISDP4_1(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #         x1, x2 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x2, x1), 1)
            
        #         x_out, log_jac_det = self.InvertISDP4_2(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=False, permute=False)
        #         x2, x1 = torch.split(x_out, self.splits, dim=1); x_out = torch.cat((x1, x2), 1)                
                   
 
        # else:
        #     x_out, log_jac_det = self.InvertISDP(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=True, permute=False)

        #     x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
        #     x_out = torch.cat((x2, x1), 1)

        #     x_out, log_jac_det = self.InvertISDP(x_out, cond_vec=c[0], logdet=log_jac_det, reverse=True, permute=False)

        #     x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
        #     x_out = torch.cat((x2, x1), 1)

        # if not rev:
        #     x_out, global_scaling_jac = self._permute(x_out, rev=False)
        # elif self.reverse_pre_permute:
        #     x_out = self._pre_permute(x_out, rev=True)

        
        # # add the global scaling Jacobian to the total.
        # # trick to get the total number of non-channel dimensions:
        # # number of elements of the first channel of the first batch member
        # n_pixels = x_out[0, :1].numel()
        # log_jac_det += (-1)**rev * n_pixels * global_scaling_jac


        # if self.cfg != None and self.cfg.use_wandb:
        #     wandb.log({'global_scale_logdet': torch.mean(global_scaling_jac), 'FlowStep_logdet': torch.mean(log_jac_det)})
    
        # if self.cfg.regularizer == 'True':
        #     regularizer = torch.norm(x_out - x[0])
        #     out_list = [x_out, regularizer]
        #     return (out_list,), log_jac_det
        return (x_out,), log_jac_det

# Flow level : [squeeze - actnorm - invertible 1x1 - affine coupling - invertible attention - split - squezze]