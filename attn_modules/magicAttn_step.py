import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import math
import wandb

class MagicISDP(nn.Module):
    def __init__(self, num_channels, split_dim, n_head, isMultiHead, cfg):
        super(MagicISDP, self).__init__()
        self.ch = num_channels 
        self.d_split = num_channels // split_dim
        # self.d_split = num_channels 
        self.split_dim = split_dim
        
        self.conv2d_q1 = nn.Conv2d(self.d_split,self.d_split,1,1) # input channel, output channel, kernel, stride
        self.conv2d_k1 = nn.Conv2d(self.d_split,self.d_split,1,1)
        self.conv2d_v1 = nn.Conv2d(self.d_split,self.d_split,1,1)
        self.s = torch.nn.Softmax(dim=-1)
        
        self.register_parameter("offset", nn.Parameter(torch.ones([1,1,1])*1.01)) # 작게라도 random init 해야할텐데 to do
        self.register_parameter("scale", nn.Parameter(torch.ones([1,1,1])*10)) # 작게라도 random init 해야할텐데 to do
        
        self.n_head = n_head
        self.isMultiHead = isMultiHead
        self.cfg = cfg
        # self.lamb = cfg.lamb
        self.register_parameter("lamb", nn.Parameter(torch.ones([1,1,1])*self.cfg.lamb))
        # self.batch_size = batch_size
        
        self.scaling_diag = nn.Parameter(torch.ones(196))
        
        if self.cfg != None and self.cfg.use_wandb:
            wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_table)
            
    def _split(self, x):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model] = [32, 196, 768]
        :return: [batch_size, head, length, d_tensor] = [32, 3, 196, 256]
        """        
        B, HW, C = x.size() # [32, 196, 768]
        dim = C // self.n_head 
        x = rearrange(x, 'b p (h d) -> b h p d', h=self.n_head, d=dim) # [32, n_head, 196, 768//n_head]
        return x
    
    def _concat(self, x):
        """
        inverse function of split
        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        B, head, HW, D = x.size()
        x = rearrange(x, 'b h p d -> b p (d h)')
        return x
            
    def forward(self, input, logdet=0, reverse=False, permute=False):
        if not reverse:
    
            p = self.ch // 2
            if self.cfg.action_type == 'toy-example':
                p_size = 1  # options: 1 or 2
            else:
                p_size = 14
            batch_size = input.shape[0] // (p_size ** 2)
            
            inp = rearrange(input, '(b h w) c -> b (h w) c', h=p_size, w=p_size) 
            
            ####### Split ####### H=14, W=14, C=768 / HW=196 <= C/n=384 : n=2,3
            qk, v = torch.split(inp, self.d_split, dim=-1) # [32, 196, ch//n] 
            qk_ = rearrange(qk, 'b (d p) c -> b c p d', d=1, p=p_size**2) 
            v = rearrange(v, 'b (d p) c -> b c p d', d=1, p=p_size**2) 

            q = self.conv2d_q1(qk_) # [32, C//n, 196, 1] 
            k = self.conv2d_k1(qk_)
            Q = rearrange(q, 'b c p d -> b p (c d)') # [32, 196, dim/C] 패치 196개의 dim 768//n_split
            K = rearrange(k, 'b c p d -> b p (c d)') 
            
            v = self.conv2d_v1(v) 
            V = rearrange(v, 'b c p d -> b p (c d)')


            if self.isMultiHead: ### Split to Multi Head ###
                # not yet
                Q, K, V = self._split(Q), self._split(K), self._split(V) # [32, n_head, 196, dim//n_head]
                attnKQ = Q @ K.transpose(2,3)  
            else:
                attnKQ = torch.bmm(Q, K.transpose(1,2))   
            

            '''case 1:''' # Attention Weight + lambda * Identity
            attnKQ= self.s(attnKQ / self.scale) # attnKQ : [32, 196, 196]
            identity = torch.eye(attnKQ.shape[-1]).to(self.cfg.device) * self.lamb
            # attnKQ = (attnKQ + identity) / (1+ self.lamb)
            attnKQ = (attnKQ + identity) / (attnKQ.shape[-1] + self.lamb) # 1-2
            
            '''case 2''' # Attention Weight + learnable diagonal
            # attnKQ= self.s(attnKQ / self.scale) # attnKQ : [32, 196, 196]
            # identity = torch.diag(torch.exp(self.scaling_diag))
            # attnKQ = attnKQ + identity
            
            # case 3: matrix_exp (Attention weight)
            # attnKQ= self.s(attnKQ / self.scale) # attnKQ : [32, 196, 196]
            # attnKQ = torch.matrix_exp(attnKQ)
            # attnKQ /= torch.sum(attnKQ)
            
        
            ### to do...
            logdet_sdp = torch.slogdet(attnKQ)[1] 
            logdet_sdp = logdet_sdp ** (-0.5)
            ### 추가적인 logdet_sdp의 스케일링?
            
            if self.isMultiHead:
                logdet_sdp = logdet_sdp.mean(dim=1)


            patch_num = p_size ** 2
            for i in range(batch_size):
                logdet[i*patch_num:(i+1)*patch_num] = logdet[i*patch_num:(i+1)*patch_num] + logdet_sdp[i]


            if self.cfg.use_wandb:
                attnRank = torch.matrix_rank(attnKQ) # attnKQ : [32, 196, 196] -> rank -> batch num
                attnRank = attnRank.type(torch.FloatTensor).mean()
                # print(attnRank)
                wandb.log({'Rank': attnRank, 'logdet': torch.mean(logdet), 'logdet_sdp': torch.mean(logdet_sdp)})

            # attention * Value
            out_attn = torch.matmul(attnKQ,V) # attnKQ+id: 32,196,196 / V: 32, 196, dim ->[32,196,dim]
            
            if self.isMultiHead: ### Concat Multi Head ###
                # not yet
                out_attn = self._concat(out_attn)
            
            out_attn = rearrange(out_attn, 'b (h w) c -> b c h w',h=p_size, w=p_size) #[32, dim, 14, 14]
            # import pdb; pdb.set_trace()
            # qk : [32, 196, ch//n] 
            # out_attn : [32, ch//n, 14, 14]
            out_attn = rearrange(out_attn, 'b c h w -> b (h w) c', h=p_size, w=p_size)
            output = torch.cat((qk, out_attn), -1) #[32, 196, 768] 
            output = rearrange(output, 'b p c-> (b p) c', p=p_size**2)


        else:
            
            ### not yet ### ver1 to 4 ... not pixed
            p = self.ch // 2
            if self.cfg.action_type == 'toy-example':
                p_size = 1  # options: 1 or 2
            else:
                p_size = 14
            batch_size = input.shape[0] // (p_size ** 2)
            # input [batch*patch num, dim] = [32*196, 768]
            out = rearrange(input, '(b h w) c -> b c h w', h=p_size, w=p_size) # [32*14*14, 768] -> [32, 768, 14, 14]
            
            ####### Split ####### H=14, W=14, C=768 / HW=196 <= C/n=384 : n=2,3
            qk, v = torch.split(out, self.d_split, dim=1) # [32, 384, 14, 14]

            q = self.conv2d_q1(qk) # [32, 768, 14, 14] -> [32, 768, 14, 14]
            k = self.conv2d_k1(qk)
            Q = rearrange(q, 'b c h w -> b (h w) c', h=p_size, w=p_size) # [32, 196, dim] 패치 196개의 dim 768//n_split
            K = rearrange(k, 'b c h w -> b (h w) c', h=p_size, w=p_size)

            # v = self.conv2d_v1(v) # for version2
            V = rearrange(v, 'b c h w -> b (h w) c', h=p_size, w=p_size) # [32, 196, dim] 패치 196개의 dim 768//n_split
 
            if self.isMultiHead: ### Split to Multi Head ###
                Q, K, V = self._split(Q), self._split(K), self._split(V) # [32, n_head, 196, dim//n_head]
                attnKQ = Q @ K.transpose(2,3)  
            else:
                attnKQ = torch.bmm(Q, K.transpose(1,2))       
            attnKQ = self.s(attnKQ / self.scale)
            # print("attnKQ : \n", attnKQ)

            attnRank = torch.matrix_rank(attnKQ) # attnKQ : [32, 196, 196] -> rank -> batch num
            attnRank = attnRank.type(torch.FloatTensor).mean()

            id = torch.eye(attnKQ.shape[-1]).cuda() * self.offset # identity matrix for logdet comptation
            logdet_sdp = torch.slogdet(attnKQ+id)[1]
            
            patch_num = p_size ** 2
            for i in range(batch_size):
                logdet[i*patch_num:(i+1)*patch_num] = logdet[i*patch_num:(i+1)*patch_num] - logdet_sdp[i]
            # logdet_sdp = torch.slogdet(attn+id)[1]*p*(p//2)*self.ch
            # logdet = logdet - logdet_sdp
            attn_inv = torch.inverse(attnKQ+id) 
            out_attn = torch.matmul(attn_inv, V)
            
            if self.isMultiHead: ### Concat Multi Head ###
                out_attn = self._concat(out_attn)
            
            out_attn = rearrange(out_attn, 'b (h w) c -> b c h w',h=p_size, w=p_size) #[32, dim, 14, 14]
            output = torch.cat((qk, out_attn), 1) #[32, 786, 14, 14] 
            output = rearrange(output, 'b c h w -> (b h w) c', h=p_size, w=p_size)
                        
        return output, logdet