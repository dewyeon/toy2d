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
from attn_modules.invertAttn_step import InvertISDP
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
        self.L = self.cfg.L_layers
        self.ActNorm = Fm.ActNorm(dims_in, dims_c)
        self.inv1x1conv = InvertibleConv1x1(dims_in[0][0], LU_decomposed=True)
        self.seed = self.cfg.seed# csflow 에서는 같은 coupling block 당 하나씩 동일시드
        if 'layer' in self.cfg.block_ver:
            self.csflow_perm = [ParallelPermute(dims_in, self.seed + i) for i in range(self.L)]
        else:
            self.csflow_perm = ParallelPermute(dims_in, self.seed) 
        self.cfg.seed += self.L
        
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels + self.L, 2 * self.splits[1])
        self.subnet2 = subnet_constructor(self.splits[0] + self.condition_channels + self.L, 2 * self.splits[1])
        if self.cfg.block_ver == 'c_s_2': # c_s_1 + affine
            self.subnet3 = subnet_constructor(self.splits[0] + self.condition_channels + self.L, 2 * self.splits[1])
            self.subnet4 = subnet_constructor(self.splits[0] + self.condition_channels + self.L, 2 * self.splits[1])


        ### OOM 안나게 필요한 모듈만 선언하기 ###
        self.resInvertISDP_1 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
        self.resInvertISDP_2 = resInvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
        
        if cfg.block_ver != 'c_only' and cfg.block_ver != 'c_only_layer' : # 이름 다시 짓기?
            self.InvertISDP_1 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
            self.InvertISDP_2 = InvertISDP(dims_in[0][0], split_dim, n_head, is_multiHead, cfg)
        
            
        if self.cfg != None and self.cfg.use_wandb:
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_table)
            
    def conditional_affine(self, x_id, x_tr, c, subnet_func, rev=False):

        if self.conditional:
            x_id_c = torch.cat([x_id, c], 1) 
        else:
            x_id_c = x_id

        if not rev:
            a1 = subnet_func(x_id_c) 
            x_tr, log_jac = self._affine(x_tr, a1)
        else:
            a1 = subnet_func(x_id_c)
            x_tr, log_jac = self._affine(x_tr, a1, rev=True)

        return x_tr, log_jac
    
    def split_resolution(self, x):
        split_res = x.shape[0] // self.L
        out_list = [[] for l in range(self.L)]
        for l in range(self.L):
            out_list[l] = x[l*split_res:(l+1)*split_res]
        return out_list
    
    def concat_resolution(self, x):
        return torch.cat(x, dim=0)
    
    def forward(self, x, c=[], rev=False, jac=True):        

        if self.cfg.block_ver == 'cs2-5': # 2-5 + csflow permutation
            '''
            0.CS-Flow Permutation for resolution-wise
            1.Cross-Attention
            2.Affine Coupling
            x1,x2 cross
            3.Cross-Attention
            4.Affine Coupling
            x1,x2 cross
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.Cross-Attention''' # x_[l]: [BHW, C], len(x_in): L
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
            log_jac_det = self.concat_resolution(log_jac_det_list) # list to vector - x: [L * BHW, C]
     
            '''2.Affine Coupling'''
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=cond_vec, subnet_func=self.subnet, rev=False)
            log_jac_det += j2

            '''x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            log_jac_det_list = self.split_resolution(log_jac_det)
        
            '''3.Cross-Attention'''
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = self.concat_resolution(x_out_list)
            log_jac_det = self.concat_resolution(log_jac_det_list)
    
            '''4.Affine Coupling'''            
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=cond_vec, subnet_func=self.subnet2, rev=False)
            log_jac_det += j1

            '''x1, x2 cross'''
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            
            out_list = self.split_resolution(x_out)
            
            return (out_list,), log_jac_det
        elif self.cfg.block_ver == 'cs2-5-2': # 2-5 + csflow permutation
            '''
            0.CS-Flow Permutation for resolution-wise
            1.Cross-Attention
            2.Affine Coupling
            x1,x2 cross
            3.Cross-Attention
            4.Affine Coupling
            x1,x2 cross
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c, dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.Cross-Attention''' # x_[l]: [BHW, C], len(x_in): L
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
            log_jac_det = self.concat_resolution(log_jac_det_list) # list to vector - x: [L * BHW, C]
     
            '''2.Affine Coupling'''
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=cond_vec, subnet_func=self.subnet, rev=False)
            log_jac_det += j1

            '''x1, x2 cross'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            log_jac_det_list = self.split_resolution(log_jac_det)
        
            '''3.Cross-Attention'''
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = self.concat_resolution(x_out_list)
            log_jac_det = self.concat_resolution(log_jac_det_list)
    
            '''4.Affine Coupling'''            
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=cond_vec, subnet_func=self.subnet2, rev=False)
            log_jac_det += j2

            '''x1, x2 cross'''
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            
            out_list = self.split_resolution(x_out)
            
            return (out_list,), log_jac_det

        elif self.cfg.block_ver == 'cs2-5_per_res': # per resolution-wise affine coupling
            '''
            0.CS-Flow Permutation for resolution-wise
            1.Cross-Attention
            2.Affine Coupling
            x1,x2 cross
            3.Cross-Attention
            4.Affine Coupling
            x1,x2 cross
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.Cross-Attention''' # x_[l]: [BHW, C], len(x_in): L
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
     
            '''2.Affine Coupling'''
            for l in range(self.L):
                x1, x2 = torch.split(x[l], self.splits, dim=1) # [L * BHW, C/2]
                x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=c[0][l], subnet_func=self.subnet, rev=False)
                log_jac_det_list[l] += j2
                
                '''x1, x2 cross'''
                x[l] = torch.cat((x2, x1), 1)

            '''3.Cross-Attention'''
            x, log_jac_det_list = self.resInvertISDP_2(x, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
    
            '''4.Affine Coupling'''
            for l in range(self.L):
                x2, x1 = torch.split(x[l], self.splits, dim=1) # [L * BHW, C/2]
                x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=c[0][l], subnet_func=self.subnet2, rev=False)
                log_jac_det_list[l] += j1
                
                '''x1, x2 cross'''
                x[l] = torch.cat((x1, x2), 1)
                
            log_jac_det = self.concat_resolution(log_jac_det_list) # list to vector - x: [L * BHW, C]
            out_list = x
            
            return (out_list,), log_jac_det
        
        elif self.cfg.block_ver == 'cs2-5-2_per_res': # per resolution-wise affine coupling
            '''
            0.CS-Flow Permutation for resolution-wise
            1.Cross-Attention
            2.Affine Coupling
            x1,x2 cross
            3.Cross-Attention
            4.Affine Coupling
            x1,x2 cross
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.Cross-Attention''' # x_[l]: [BHW, C], len(x_in): L
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
     
            '''2.Affine Coupling'''
            for l in range(self.L):
                x1, x2 = torch.split(x[l], self.splits, dim=1) # [L * BHW, C/2]
                x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=c[0][l], subnet_func=self.subnet, rev=False)
                log_jac_det_list[l] += j1
                
                '''x1, x2 cross'''
                x[l] = torch.cat((x2, x1), 1)

            '''3.Cross-Attention'''
            x, log_jac_det_list = self.resInvertISDP_2(x, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
    
            '''4.Affine Coupling'''
            for l in range(self.L):
                x2, x1 = torch.split(x[l], self.splits, dim=1) # [L * BHW, C/2]
                x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=c[0][l], subnet_func=self.subnet2, rev=False)
                log_jac_det_list[l] += j2
                
                '''x1, x2 cross'''
                x[l] = torch.cat((x1, x2), 1)
                
            log_jac_det = self.concat_resolution(log_jac_det_list) # list to vector - x: [L * BHW, C]
            out_list = x
            
            return (out_list,), log_jac_det

        elif self.cfg.block_ver == 'c_s_1':
            '''
            0.CS-Flow Permutation for resolution-wise
            1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]
            2.Affine Coupling # split 하고 concat할때 x1,x2 cross조합  잘하기
            3.[Self-Attention] - [x1,x2 cross] - [Self-Attention]
            4.Affine Coupling # split 하고 concat할때 x1,x2 cross조합 잘하기
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]'''
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
     
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = self.concat_resolution(x_out_list)
            log_jac_det = self.concat_resolution(log_jac_det_list)

            '''2.Affine Coupling''' # 여기서는 x2 transformation
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=cond_vec, subnet_func=self.subnet, rev=False)
            log_jac_det += j1
            
            '''3.[Self-Attention] - [x1,x2 cross] - [Self-Attention]'''
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_1(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_2(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)

            '''4.Affine Coupling''' # 여기서는 x1 transformation
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=cond_vec, subnet_func=self.subnet2, rev=False)
            log_jac_det += j2

            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            out_list = self.split_resolution(x_out)
            
            return (out_list,), log_jac_det

        elif self.cfg.block_ver == 'c_s_2': # c_s_1 + affine
            '''
            0.CS-Flow Permutation for resolution-wise
            1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]
            2.Affine Coupling Twice for both x1,x2
            3.[Self-Attention] - [x1,x2 cross] - [Self-Attention]
            4.Affine Coupling Twice for both x1,x2
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]'''
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
     
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = self.concat_resolution(x_out_list)
            log_jac_det = self.concat_resolution(log_jac_det_list)

            '''2.Affine Coupling for Twice'''
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=cond_vec, subnet_func=self.subnet, rev=False)
            log_jac_det += j1
            x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=cond_vec, subnet_func=self.subnet2, rev=False)
            log_jac_det += j2                       
            
            '''3.[Self-Attention] - [x1,x2 cross] - [Self-Attention]'''
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_1(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_2(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)

            '''4.Affine Coupling''' # 여기서는 x1 transformation
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [BHW, C/2]
            x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=cond_vec, subnet_func=self.subnet3, rev=False)
            log_jac_det += j1
            x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=cond_vec, subnet_func=self.subnet4, rev=False)
            log_jac_det += j2

            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            out_list = self.split_resolution(x_out)
            
            return (out_list,), log_jac_det
        
        elif self.cfg.block_ver == 'c_only': 
            '''
            0.CS-Flow Permutation for resolution-wise
            1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]'''
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
     
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            log_jac_det = self.concat_resolution(log_jac_det_list)

            out_list = x_out_list
            
            return (out_list,), log_jac_det

        elif self.cfg.block_ver == 'cs': 
            '''
            0.CS-Flow Permutation for resolution-wise
            1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]
            2.[Self-Attention] - [x1,x2 cross] - [Self-Attention]
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]'''
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
     
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = self.concat_resolution(x_out_list)
            log_jac_det = self.concat_resolution(log_jac_det_list)

            
            '''2.[Self-Attention] - [x1,x2 cross] - [Self-Attention]'''
            # x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_1(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_2(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)

  
            out_list = self.split_resolution(x_out)
            
            return (out_list,), log_jac_det
        
        elif self.cfg.block_ver == 'cs2': 
            '''
            0.CS-Flow Permutation for resolution-wise
            1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]
            2. Affine Couling Twice for x1,x2
            3.[Self-Attention] - [x1,x2 cross] - [Self-Attention]
            '''
            log_jac_det_list = [0 for l in range(self.L)] # store for per resolution
            x_in = [0 for l in range(self.L)] # store for per resolution
            cond_vec = torch.cat(c[0], dim=0)
            
            '''0.CS-Flow Permutation for resolution-wise'''
            for l in range(self.L):
                if 'layer' in self.cfg.block_ver:
                    x_in[l] = self.csflow_perm[l](x[0][l], rev=rev)
                else:
                    x_in[l] = self.csflow_perm(x[0][l], rev=rev)
              
            '''1.[Cross-Attention] - [x1,x2 cross] - [Cross-Attention]'''
            x, log_jac_det_list = self.resInvertISDP_1(x_in, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x = self.concat_resolution(x) # list to vector - x: [L * BHW, C]
     
            x1, x2 = torch.split(x, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]

            out_list = self.split_resolution(x_out)
            x_out_list, log_jac_det_list = self.resInvertISDP_2(out_list, cond_vec=c[0], logdet=log_jac_det_list, reverse=False, permute=False)
            x_out = self.concat_resolution(x_out_list)
            log_jac_det = self.concat_resolution(log_jac_det_list)

            '''2.Affine Coupling for Twice'''
            x2, x1 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x1, j1 = self.conditional_affine(x_id=x2, x_tr=x1, c=cond_vec, subnet_func=self.subnet, rev=False)
            log_jac_det += j1
            x2, j2 = self.conditional_affine(x_id=x1, x_tr=x2, c=cond_vec, subnet_func=self.subnet2, rev=False)
            log_jac_det += j2    
            
            '''3.[Self-Attention] - [x1,x2 cross] - [Self-Attention]'''
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_1(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)
            x1, x2 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x2, x1), 1) # [BHW, C]
            x_out, log_jac_det = self.InvertISDP_2(x_out, cond_vec=cond_vec, logdet=log_jac_det, reverse=False, permute=False)

            x2, x1 = torch.split(x_out, self.splits, dim=1) # [L * BHW, C/2]
            x_out = torch.cat((x1, x2), 1) # [BHW, C]
  
            out_list = self.split_resolution(x_out)
            
            return (out_list,), log_jac_det
        

        
        
        
        
        
# Flow level : [squeeze - actnorm - invertible 1x1 - affine coupling - invertible attention - split - squezze]