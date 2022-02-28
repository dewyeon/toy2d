import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import math

class ISDP(nn.Module):
    def __init__(self, num_channels, cfg, n_head, isMultiHead):
        super(ISDP, self).__init__()
        self.ch = num_channels 

        self.conv2d_q1 = nn.Conv2d(num_channels,num_channels,1,1) # input channel, output channel, kernel, stride
        self.conv2d_k1 = nn.Conv2d(num_channels,num_channels,1,1)
        self.s = torch.nn.Softmax(dim=-1)
        
        self.register_parameter("offset", nn.Parameter(torch.ones([1,1,1])*1.01)) # 작게라도 random init 해야할텐데 to do
        self.register_parameter("scale", nn.Parameter(torch.ones([1,1,1])*1000)) # 작게라도 random init 해야할텐데 to do
        
        self.n_head = n_head
        self.isMultiHead = isMultiHead
        self.cfg = cfg
        # self.batch_size = batch_size

    # function for checkerboard masking
    def _create_HW_checkboard(self, batch, size, channel, reverse=False):
        x_, y_ = torch.arange(size, dtype=torch.int32), torch.arange(size, dtype=torch.int32)
        x, y = torch.meshgrid(x_, y_)
        mask = torch.fmod(x+y, 2)
        mask_a = mask.to(torch.float32).view(1, 1, size, size)
        mask_b = 1 - mask_a
        if reverse:
            mask_b = mask_a
            mask_a = mask_b
        mask_set = torch.cat((mask_a, mask_b), dim=1)
        mask_hw_wise = mask_set.repeat(batch, channel//2, 1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_hw_wise = mask_hw_wise.to(device)
        return mask_hw_wise

    def _create_ch_checkboard(self, batch, size, channel, reverse=False):
        x_, y_ = torch.arange(channel, dtype=torch.int32), torch.arange(size, dtype=torch.int32)
        x, y = torch.meshgrid(x_, y_)
        mask = torch.fmod(x+y, 2)
        mask_a = mask.to(torch.float32).view(1, channel, size, 1)
        mask_b = 1 - mask_a
        if reverse:
            mask_b = mask_a
            mask_a = mask_b
        mask_set = torch.cat((mask_a, mask_b), dim=-1) # 1, 768 14, 2
        mask_ch_wise = mask_set.repeat(batch, 1, 1, size // 2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_ch_wise = mask_ch_wise.to(device)
        return mask_ch_wise
    
    def _checkboard(self, batch, H, W, C, reverse=False):
        x_, y_ = torch.arange(H*W, dtype=torch.int32), torch.arange(C, dtype=torch.int32)
        x, y = torch.meshgrid(x_, y_)
        m = torch.fmod(x+y, 2) # 196,768
        mask_a = m.to(torch.float32)
        mask_a = rearrange(mask_a, '(d p) c -> d p c', d=1)
        mask_b = 1 - mask_a
        if reverse:
            mask_b = mask_a
            mask_a = mask_b
        mask_set = torch.cat((mask_a, mask_b), dim=0) # 1, 768 14, 2
        mask_wise = mask_set.repeat(batch // 2, 1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_wise = mask_wise.to(device)
    
        return mask_wise

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
        # ori_dim = head * D
        # x = x.transpose(1,2).contiguous().view(B, HW, ori_dim)
        return x
        

    def forward(self, input, logdet=0, reverse=False, permute=False):
        if not reverse:
            # # ViT patch [batch, 196, 768]
            # input [batch*patch, 768] [6272, 768]
            # # transpose하면서 patch단위가 아니라 Feature단위로 계산되고 있는듯
            # p = input.shape[-1]
            # import pdb; pdb.set_trace()
            p = self.ch // 2
            if self.cfg.action_type == 'toy-example':
                p_size = 1  # options: 1 or 2
            else:
                p_size = 14
            batch_size = input.shape[0] // (p_size ** 2)
            
            mask_hw = self._create_HW_checkboard(batch_size, p_size, self.ch, reverse=reverse)
            mask_ch = self._create_ch_checkboard(batch_size, p_size, self.ch, reverse=reverse)
            mask = mask_hw 
         
            inp = rearrange(input, '(b h w) c -> b c h w', h=p_size, w=p_size) # [32*14*14, 768] -> [32, 768, 14, 14]
            qk = inp * mask
            v = inp * (1-mask) # for value
            V = rearrange(v, 'b c h w -> b (h w) c', h=p_size, w=p_size) # [32, 196, 768] 각 Dimension이 768인 패치 196개

            q = self.conv2d_q1(qk) # [32, 768, 14, 14] -> [32, 768, 14, 14]
            k = self.conv2d_k1(qk)
            
            Q = rearrange(q, 'b c h w -> b (h w) c', h=p_size, w=p_size) # [32, 196, 768] 각 Dimension이 768인 패치 196개
            K = rearrange(k, 'b c h w -> b (h w) c', h=p_size, w=p_size) 
            
            if self.isMultiHead: ### Split to Multi Head ###
                Q, K, V = self._split(Q), self._split(K), self._split(V) # [32, n_head, 196, 768//n_head]
                attnKQ = Q @ K.transpose(2,3)  
            else:
                attnKQ = torch.bmm(Q, K.transpose(1,2))    
            attnKQ = self.s(attnKQ / self.scale) # scaling random init
            # print("attnKQ : \n", attnKQ)
            
            id = torch.eye(attnKQ.shape[-1]).cuda() * self.offset
            
            # 수식을 갖고            
            logdet_sdp = torch.slogdet(attnKQ + id)[1] # identity 값을 어떻게 더해주는지.. epsilon을 더해주는지 1을 더해주는지
            # logdet_sdp = torch.slogdet(attnKQ + id)[1] * p_size * (p_size //2 ) / self.ch
            
            # 기존 Pseudo code
            # logdet_sdp = torch.slogdet(attnKQ + id)[1] * p_size * (p_size//2) * self.ch
            
            # logdet_sdp = logdet_sdp/ logdet_sdp.mean()
            # import pdb; pdb.set_trace()
            if self.isMultiHead:
                logdet_sdp = logdet_sdp.mean(dim=1)
            
            patch_num = p_size ** 2
            for i in range(batch_size):
                logdet[i*patch_num:(i+1)*patch_num] = logdet[i*patch_num:(i+1)*patch_num] + logdet_sdp[i]
    
            out_attn = torch.matmul(attnKQ+id,V) # attnKQ+id: 32,196,196 / V: 32, 196, 768
            
            if self.isMultiHead: ### Concat Multi Head ###
                out_attn = self._concat(out_attn)
                
            out_attn = rearrange(out_attn, 'b (h w) c -> b c h w',h=p_size, w=p_size)
            output = out_attn * (1-mask) + inp * mask
            output = rearrange(output, 'b c h w -> (b h w) c', h=p_size, w=p_size)
        else:
            p = self.ch // 2
            if self.cfg.action_type == 'toy-example':
                p_size = 1  # options: 1 or 2
            else:
                p_size = 14
            batch_size = input.shape[0] // (p_size ** 2)
            # input [batch*patch num, dim] = [32*196, 768]
            out = rearrange(input, '(b h w) c -> b c h w', h=p_size, w=p_size) # [32, 768, 14, 14]
                        
            mask_hw = self._create_HW_checkboard(batch_size, p_size, self.ch, reverse=True)
            mask = mask_hw 
            # import pdb; pdb.set_trace()()
            
            if permute:
                mask = 1 - mask

            qk = out * mask
            v = out * (1-mask)
            V = rearrange(v, 'b c h w -> b (h w) c', h=p_size, w=p_size) 

            q = self.conv2d_q1(qk) # [32, 768, 14, 14] -> [32, 768, 14, 14]
            k = self.conv2d_k1(qk)
             
            Q = rearrange(q, 'b c h w -> b (h w) c', h=p_size, w=p_size) 
            K = rearrange(k, 'b c h w -> b (h w) c', h=p_size, w=p_size)    
            
            if self.isMultiHead: ### Split to Multi Head ###
                Q, K, V = self._split(Q), self._split(K), self._split(V) # [32, n_head, 196, 768//n_head]
                attnKQ = Q @ K.transpose(2,3)  
            else:
                attnKQ = torch.bmm(Q, K.transpose(1,2))       
            attnKQ = self.s(attnKQ / self.scale)
            # print("attnKQ : \n", attnKQ)

            id = torch.eye(attnKQ.shape[-1]).cuda() * self.offset # identity matrix for logdet comptation
            logdet_sdp = torch.slogdet(attnKQ+id)[1]
            patch_num = p_size ** 2
            for i in range(batch_size):
                logdet[i*patch_num:(i+1)*patch_num] = logdet[i*patch_num:(i+1)*patch_num] - logdet_sdp[i]
            # logdet_sdp = torch.slogdet(attn+id)[1]*p*(p//2)*self.ch
            # logdet = logdet - logdet_sdp
            # import pdb; pdb.set_trace()()
            attn_inv = torch.inverse(attnKQ+id) 
            out_attn = torch.matmul(attn_inv, V)
            
            if self.isMultiHead: ### Concat Multi Head ###
                out_attn = self._concat(out_attn)
            
            out_attn = rearrange(out_attn, 'b (h w) c -> b c h w',h=p_size, w=p_size)
            output = out_attn * (1-mask) + out * mask
            output = rearrange(output, 'b c h w -> (b h w) c', h=p_size, w=p_size)

        return output, logdet