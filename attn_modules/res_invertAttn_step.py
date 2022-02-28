import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import math
import wandb


class resInvertISDP(nn.Module):
    def __init__(self, num_channels, split_dim, n_head, isMultiHead, cfg):
        super(resInvertISDP, self).__init__()
        self.ch = num_channels 
        self.d_split = num_channels // split_dim
        self.attn_dim = cfg.conv_attn # d'
        self.split_dim = split_dim

        self.s = torch.nn.Softmax(dim=-1)
        self.cfg = cfg
        self.L = cfg.L_layers

        if self.cfg.action_type == 'toy-example':
            self.H = self.cfg.toy_H
            self.W = self.cfg.toy_W
        else:
            if 'vit' in self.cfg.enc_arch:
                if 'patch16_224' in self.cfg.enc_arch:
                    self.H, self.W = 14, 14
                elif 'patch32_224' in self.cfg.enc_arch:
                    self.H, self.W = 7, 7
                elif 'patch8_224' in self.cfg.enc_arch:
                    self.H, self.W = 28, 28
                elif 'patch16_384' in self.cfg.enc_arch:
                    self.H, self.W = 24, 24                
                elif 'patch32_384' in self.cfg.enc_arch:
                    self.H, self.W = 12, 12
            elif 'deit' in self.cfg.enc_arch:
                self.H, self.W = 14, 14
            elif 'cait' in self.cfg.enc_arch:
                if '224' in self.cfg.enc_arch:
                    self.H, self.W = 14, 14
                elif '384' in self.cfg.enc_arch:
                    self.H, self.W = 24, 24
            elif 'swin' in self.cfg.enc_arch:
                if '224' in self.cfg.enc_arch:
                    self.H, self.W = 14, 14
                elif '384' in self.cfg.enc_arch:
                    self.H, self.W = 24, 24
            elif 'resnet' in self.cfg.enc_arch:
                self.H, self.W = 8, 8
            
        self.qkv_dim = self.d_split + self.cfg.condition_vec + self.cfg.L_layers +  1 # [c/2 + pos_enc + L_layers + 1] 
        self.s = torch.nn.Softmax(dim=-1)

        ### Y' ###
        conv2d_Wq = nn.Conv2d(self.qkv_dim, self.attn_dim, 1, 1, bias=False) # input channel, output channel, kernel, stride
        conv2d_Wk = nn.Conv2d(self.qkv_dim, self.attn_dim, 1, 1, bias=False) # input channel, output channel, kernel, stride
        conv2d_Wv = nn.Conv2d(self.qkv_dim, self.attn_dim, 1, 1, bias=False) # input channel, output channel, kernel, stride

        wq = nn.Parameter(torch.Tensor(self.attn_dim, self.qkv_dim, 1, 1))
        wk = nn.Parameter(torch.Tensor(self.attn_dim, self.qkv_dim, 1, 1))
        wv = nn.Parameter(torch.Tensor(self.attn_dim, self.qkv_dim, 1, 1))

        nn.init.xavier_uniform_(wq);nn.init.xavier_uniform_(wk);nn.init.xavier_uniform_(wv)
        # nn.init.constant_(wv1, 0.)
        wv.data = wv.data * 0.01;wv.data[:, -1] = 1.0
        conv2d_Wq.weight.data = wq;conv2d_Wk.weight.data = wk;conv2d_Wv.weight.data = wv
        
        ### Y'' ###
        conv2d_Wq2 = nn.Conv2d(self.qkv_dim, self.attn_dim, 1, 1, bias=False) # input channel, output channel, kernel, stride
        conv2d_Wk2 = nn.Conv2d(self.qkv_dim, self.attn_dim, 1, 1, bias=False) # input channel, output channel, kernel, stride
        conv2d_Wv2 = nn.Conv2d(self.qkv_dim, self.attn_dim, 1, 1, bias=False) # input channel, output channel, kernel, stride

        wq2 = nn.Parameter(torch.Tensor(self.attn_dim, self.qkv_dim, 1, 1))
        wk2 = nn.Parameter(torch.Tensor(self.attn_dim, self.qkv_dim, 1, 1))
        wv2 = nn.Parameter(torch.Tensor(self.attn_dim, self.qkv_dim, 1, 1))

        nn.init.xavier_uniform_(wq2);nn.init.xavier_uniform_(wk2);nn.init.xavier_uniform_(wv2)
        # nn.init.constant_(wv1, 0.)
        wv2.data = wv2.data * 0.01;wv2.data[:, -1] = 1.0
        conv2d_Wq2.weight.data = wq2;conv2d_Wk2.weight.data = wk2;conv2d_Wv2.weight.data = wv2
        
        ''' A, B is matrix '''
        W_A = nn.Conv2d(self.attn_dim, self.d_split, 1, 1, bias=False) # [d', C/2]
        W_B = nn.Conv2d(self.attn_dim, self.d_split, 1, 1, bias=False) # [d', C/2]
        wa = nn.Parameter(torch.Tensor(self.d_split, self.attn_dim, 1, 1))
        wb = nn.Parameter(torch.Tensor(self.d_split, self.attn_dim, 1, 1))
        nn.init.xavier_uniform_(wa)
        wa.data = wa.data * 0.01
        nn.init.xavier_uniform_(wb)
        wa.data[:, 0] = 1.0
        W_A.weight.data = wa
        W_B.weight.data = wb
        # offset = nn.Parameter(torch.ones([1,1,1])*1.01)
        self.scale1 = nn.Parameter(torch.ones([1,1,1])*self.cfg.sa_scale1)
        self.scale2 = nn.Parameter(torch.ones([1,1,1])*self.cfg.sa_scale2)

        if self.L >= 2:
            self.scale1_2 = nn.Parameter(torch.ones([1,1,1])*self.cfg.sa_scale1)
            self.scale2_2 = nn.Parameter(torch.ones([1,1,1])*self.cfg.sa_scale2)
        if self.L >= 3:
            self.scale1_3 = nn.Parameter(torch.ones([1,1,1])*self.cfg.sa_scale1)
            self.scale2_3 = nn.Parameter(torch.ones([1,1,1])*self.cfg.sa_scale2)

        qkv_module = nn.ModuleList([conv2d_Wq, conv2d_Wk, conv2d_Wv, W_A, conv2d_Wq2, conv2d_Wk2, conv2d_Wv2, W_B])        
        
        self.qkv_modules = nn.ModuleList([])
        for l in range(self.L):
            self.qkv_modules.append(qkv_module)

        # self.register_parameter("paramA0", nn.Parameter(torch.zeros([1,1,1]))) # 작게라도 random init 해야할텐데 to do
        # self.register_parameter("paramA1", nn.Parameter(torch.ones([1,1,1]))) # 작게라도 random init 해야할텐데 to do
        # self.register_parameter("paramB", nn.Parameter(torch.zeros([1,1,1]))) # 작게라도 random init 해야할텐데 to do


        self.n_head = n_head
        self.isMultiHead = isMultiHead
        self.cfg = cfg
        self.register_parameter("lamb", nn.Parameter(torch.ones([1,1,1])*self.cfg.lamb))

    def _split(self, x):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model] = [32, 196, 768]
        :return: [batch_size, head, length, d_tensor] = [32, 3, 196, 256]
        """        
        B, HW, C = x.size() 
        dim = C // self.n_head 
        x = rearrange(x, 'b p (h d) -> b h p d', h=self.n_head, d=dim) # [B, n_head, HW, C/2/n_head]
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
            
    def forward(self, inputs, cond_vec, logdet, reverse=False, permute=False):
        if not reverse:
            x_id, x_tr = [[] for l in range(self.L)], [[] for l in range(self.L)] # inputs: layer0: inputs[0], layer1: inputs[1], layer2: inputs[2]
            y = [[] for l in range(self.L)]
            output_list = [[] for l in range(self.L)]
            wandb_log = {}
            ####### Split ####### H=14, W=14, C=768 / HW=196 <= C/n=384 : n=2
            for l in range(self.L):
                inp = rearrange(inputs[l], '(b h w) c -> b (h w) c', h=self.H, w=self.W) 
                x_id[l], x_tr[l] = torch.split(inp, self.d_split, dim=-1)
                DC = torch.ones(x_id[l].shape[0], x_id[l].shape[1], 1).to(self.cfg.device)
                cond = rearrange(cond_vec[l], '(b hw) d -> b hw d', b=x_id[l].shape[0])
                y[l] = torch.cat((x_id[l], cond, DC), dim=-1)


            if self.L == 1 or self.L==3:
                print("not implemented")
                return
            
            for idx_q in range(self.L):

                if self.L == 2:
                    if idx_q == 0: 
                        idx_kv = 1
                        scale1 = self.scale1
                        scale2 = self.scale2
                    else: 
                        idx_kv = 0
                        scale1 = self.scale1_2
                        scale2 = self.scale2_2

                    q, kv = rearrange(y[idx_q], 'b (h w) k -> b k h w', h=self.H, w=self.W), rearrange(y[idx_kv], 'b (h w) k -> b k h w', h=self.H, w=self.W)
                    
                    q1, k1, v1 = self.qkv_modules[idx_q][0](q), self.qkv_modules[idx_q][1](kv), self.qkv_modules[idx_q][2](kv)
                    q2, k2, v2 = self.qkv_modules[idx_q][4](q), self.qkv_modules[idx_q][5](kv), self.qkv_modules[idx_q][6](kv)
                    Q1, K1, V1 = rearrange(q1, 'b k h w -> b (h w) k'), rearrange(k1, 'b k h w -> b (h w) k'), rearrange(v1, 'b k h w -> b (h w) k')
                    Q2, K2, V2 = rearrange(q2, 'b k h w -> b (h w) k'), rearrange(k2, 'b k h w -> b (h w) k'), rearrange(v2, 'b k h w -> b (h w) k')
                
                    Q_Kt1 = self.s((torch.bmm(Q1, K1.transpose(1,2))) / scale1)
                    SA1 = torch.bmm(Q_Kt1,V1) # SA(y) : [B, HW 15, conv_attn]
                    SA1 = rearrange(SA1, 'b (h w) d -> b d h w', h=self.H, w=self.W)

                    Q_Kt2 = self.s((torch.bmm(Q2, K2.transpose(1,2))) / scale2)
                    SA2 = torch.bmm(Q_Kt2,V2)  
                    SA2 = rearrange(SA2, 'b (h w) d -> b d h w', h=self.H, w=self.W)
                    A, B = self.qkv_modules[idx_q][3](SA1), self.qkv_modules[idx_q][7](SA2) 
                    A, B = rearrange(A, 'b d h w -> b (h w) d'), rearrange(B, 'b d h w -> b (h w) d')
                    
                    transformation = self.cfg.trans_ver[-1]
                    operator = int(self.cfg.trans_ver[0]) % 2
                    
                    if transformation == '1':
                        A = torch.exp(A)
                    elif transformation == '2':
                        A = torch.exp(A)
                        B = torch.exp(B)
                    elif transformation == '3':
                        A = torch.sigmoid(A+2)

                    if operator: # case: 2m-1. X'=AX+B
                        half_out = A * x_tr[idx_q] + B
                    else:
                        half_out = A * (x_tr[idx_q] + B)
                
                    '''log determinant Jacobian'''
                    logdet_SA = torch.log(torch.abs(A))
                    logdet_SA = torch.sum(logdet_SA, dim=-1) 
                    logdet_SA = rearrange(logdet_SA, 'b p -> (b p)')            
                    
                    logdet[idx_q] += logdet_SA
     
                    output = torch.cat((x_id[idx_q], half_out), -1) # [B, HW, C]
                    output = rearrange(output, 'b p c -> (b p) c', p=self.H*self.W)
                    output_list[idx_q] = output
                    
                

            # if self.cfg.use_wandb:
            #     wandb.log({'logdet': torch.mean(logdet), 'logdet_SA': torch.mean(logdet_SA),
            #                 'A': A.mean(), 'B': B.mean(), 'half_out':half_out.mean(),
            #                 # 'weight_A': self.W_A.weight.mean(), 'weight_B': self.W_B.weight.mean(),
            #                 # 'weight_Wv_A': self.conv2d_Wv1.weight.mean(), 'weight_Wv_B': self.conv2d_Wv2.weight.mean(),
            #                 'SA1': SA1.mean(), 'SA2': SA2.mean(),
            #             })
    
        else:
            out = rearrange(inputs, '(b h w) c -> b (h w) c', h=self.H, w=self.W) 

            ####### Split ####### H=14, W=14, C=768 / HW=196 <= C/n=384 : n=2,
            y_tilde, x_tilde = torch.split(out, self.d_split, dim=-1)
            DC = torch.ones(y_tilde.shape[0], y_tilde.shape[1], 1).to(self.cfg.device)
            cond_vec = rearrange(cond_vec, '(b p) d -> b p d', b=y_tilde.shape[0])
            y = torch.cat((y_tilde, cond_vec, DC), dim=-1) # [1, 15, 6]

            ####### Self-attention with y_identity #######
            qkv = rearrange(y, 'b (d p) c -> b c p d', d=1, p=self.H*self.W) # [B, C, HW, 1] add one dim for convolution
            q1,k1 = self.conv2d_Wq1(qkv), self.conv2d_Wk1(qkv) # [B, C, HW, 1]
            q2,k2 = self.conv2d_Wq2(qkv), self.conv2d_Wk2(qkv) # [B, C, HW, 1]
            
            v1 = self.conv2d_Wv1(qkv)
            v2 = self.conv2d_Wv2(qkv)
            V1 = rearrange(v1, 'b c p d -> b p (c d)')
            V2 = rearrange(v2, 'b c p d -> b p (c d)')

            # v1 = y @ self.conv2d_Wv1 
            # v2 = y @ self.conv2d_Wv2
            # V1, V2 = v1, v2

            Q1, K1 = rearrange(q1, 'b c p d -> b p (c d)'), rearrange(k1, 'b c p d -> b p (c d)')
            Q2, K2 = rearrange(q2, 'b c p d -> b p (c d)'), rearrange(k2, 'b c p d -> b p (c d)')


            if self.isMultiHead: ### Split to Multi Head ###
                Q1, K1, V1 = self._split(Q1), self._split(K1), self._split(V1) # [32, n_head, 196, C/2/n_head]
                Q2, K2, V2 = self._split(Q2), self._split(K2), self._split(V2) # [32, n_head, 196, C/2/n_head]
                
                Q_Kt1 = self.s(Q1 @ K1.transpose(2,3) / self.scale1)
                Q_Kt2 = self.s(Q2 @ K2.transpose(2,3) / self.scale2)

                SA1 = torch.matmul(Q_Kt1,V1)
                SA2 = torch.matmul(Q_Kt2,V2)

                import pdb; pdb.set_trace()
                # todo
            else:
                Q_Kt1 = self.s((torch.bmm(Q1, K1.transpose(1,2))) / self.scale1)
                SA1 = torch.bmm(Q_Kt1,V1)

                Q_Kt2 = self.s((torch.bmm(Q2, K2.transpose(1,2))) / self.scale2)
                SA2 = torch.bmm(Q_Kt2,V2)

                # A = SA1 @ self.W_A #'''for toy-test 1 '''                
                # B = SA2 @ self.W_B #'''for toy-test 1 '''
                A = self.W_A(SA1) # '''for toy-test 2 : optimization '''
                B = self.W_B(SA2) # '''for toy-test 2 : optimization '''     

                half_out = (x_tilde - B) / A
            

            if self.cfg.action_type == 'toy-example':
                print("A: ", A.mean(), "\t SA1:", SA1.mean())
                print("B: ", B.mean(), "\t SA2:", SA2.mean())

            '''log determinant Jacobian'''
            I = torch.ones(self.d_split).to(self.cfg.device)
            # logdet_SA = torch.log(torch.abs(A + I))
            logdet_SA = torch.log(torch.abs(A))
            logdet_SA = torch.sum(logdet_SA, dim=-1) * self.d_split
            logdet_SA = rearrange(logdet_SA, 'b p -> (b p)')    
            

            if self.isMultiHead:
                logdet_SA = logdet_SA.mean(dim=1)

            logdet = logdet - logdet_SA
            
            if self.isMultiHead: ### Concat Multi Head ###
                aff_out = self._concat(half_out)


            output = torch.cat((y_tilde, half_out), -1) # [B, HW, C]
            output = rearrange(output, 'b p c -> (b p) c', p=self.H*self.W)
                  
        return output_list, logdet



# class InvertISDP(nn.Module):
# #     def __init__(self, num_channels, split_dim, n_head, isMultiHead, cfg):
# #         super(InvertISDP, self).__init__()
# #         self.ch = num_channels
# #         self.d_split = num_channels // split_dim
# #         self.attention_dim = cfg.toy_attention_dim
# #         self.H = cfg.toy_H
# #         self.W = cfg.toy_W
# #         # self.W_a_mat = nn.Parameter(torch``.ones([self.attention_dim, self.d_split])) # [d', C/2]
# #         # self.W_b_mat = nn.Parameter(torch.ones([self.attention_dim, self.d_split])) # [d', C/2]
# #         # import pdb; pdb.set_trace()

# #         self.W_a_mat = torch.zeros(self.attention_dim, self.d_split).unsqueeze(0)
# #         self.W_a_mat[0][0][0] = 0.5
# #         self.W_b_mat = torch.zeros(self.attention_dim, self.d_split).unsqueeze(0)
# #         self.W_b_mat[0][0][0] = -0.5
# #         self.split_dim = split_dim


# #         self.d_pos_enc = cfg.toy_pos_enc
# #         self.conv2d_q1 = nn.Conv2d(self.d_split + self.d_pos_enc + 1, self.attention_dim, 1, 1) # input channel, output channel, kernel, stride
# #         self.conv2d_k1 = nn.Conv2d(self.d_split + self.d_pos_enc + 1, self.attention_dim, 1, 1)
# #         # self.conv2d_v1 = nn.Conv2d(self.d_split + self.d_pos_enc + 1, self.attention_dim, 1, 1)
# #         self.conv2d_v1 = torch.zeros((self.d_split + self.d_pos_enc + 1), self.attention_dim).unsqueeze(0)
# #         self.conv2d_v1[0][-1][0] = 1


# #         self.s = torch.nn.Softmax(dim=-1)
# #         # self.register_parameter("offset", nn.Parameter(torch.ones([1,1,1])*1.01)) # 작게라도 random init 해야할텐데 to do
# #         self.register_parameter("scale", nn.Parameter(torch.ones([1,1,1])*10)) # 작게라도 random init 해야할텐데 to do
# #         self.n_head = n_head
# #         self.isMultiHead = isMultiHead
# #         self.cfg = cfg
# #         self.register_parameter("lamb", nn.Parameter(torch.ones([1,1,1])*self.cfg.lamb))
# #         # self.batch_size = batch_size
# #         if self.cfg != None and self.cfg.use_wandb:
# #             wandb.init(project=self.cfg.wandb_project, name=self.cfg.wandb_table)
# #     def _split(self, x):
# #         """
# #         split tensor by number of head
# #         :param tensor: [batch_size, length, d_model] = [32, 196, 768]
# #         :return: [batch_size, head, length, d_tensor] = [32, 3, 196, 256]
# #         """
# #         B, HW, C = x.size()
# #         dim = C // self.n_head
# #         x = rearrange(x, 'b p (h d) -> b h p d', h=self.n_head, d=dim) # [B, n_head, HW, C/2/n_head]
# #         return x
# #     def _concat(self, x):
# #         """
# #         inverse function of split
# #         :param tensor: [batch_size, head, length, d_tensor]
# #         :return: [batch_size, length, d_model]
# #         """
# #         B, head, HW, D = x.size()
# #         x = rearrange(x, 'b h p d -> b p (d h)')
# #         return x
# #     def forward(self, input, pos_enc, logdet=0, reverse=False, permute=False):
# #         if not reverse:
# #             p = self.ch // 2
# #             if self.cfg.action_type == 'toy-example':
# #                 p_size = 1  # options: 1 or 2
# #             else:
# #                 p_size = 14
# #             batch_size = input.shape[0] // (p_size ** 2)
# #             inp = rearrange(input, '(b h w) c -> b (h w) c', h=self.H, w=self.W)
# #             # import pdb; pdb.set_trace()
# #             ####### 1. Split X ####### H=14, W=14, C=768 / HW=196 <= C/n=384 : n=2
# #             x_tilde, y_tilde = torch.split(inp, self.d_split, dim=-1) # x_identity(self-attention), x_transformation [32, HW196, ch//n]
# #             ####### 2. Make Y #######
# #             pos_enc = pos_enc.unsqueeze(0)
# #             ones = torch.ones(y_tilde.shape[1]).unsqueeze(0).unsqueeze(-1)
# #             y = torch.cat((y_tilde, pos_enc, ones), dim=2)
# #             ####### 3. Self-attention: SA(Y) #######
# #             # qkv = rearrange(y, 'b (d p) c -> b c p d', d=1, p=p_size**2) # [B, HW, C/2, 1] add one dim for convolution
# #             qkv = rearrange(y, 'b (hw d) c -> b c hw d', d=1)
# #             # import pdb; pdb.set_trace()
# #             q, k = self.conv2d_q1(qkv), self.conv2d_k1(qkv)# # [B, C/2, HW, 1]
# #             v = y @ self.conv2d_v1
# #             Q = rearrange(q, 'b c p d -> b p (c d)')
# #             K = rearrange(k, 'b c p d -> b p (c d)')
# #             # V = rearrange(v, 'b c p d -> b p (c d)') # [b, HW, C/2]
# #             V = v
# #             if self.isMultiHead: ### Split to Multi Head ###
# #                 Q, K, V = self._split(Q), self._split(K), self._split(V) # [B, n_head, 196, C/2/n_head]
# #                 Q_Kt = Q @ K.transpose(2,3)
# #                 Q_Kt = self.s(Q_Kt / self.scale) # [B, Head, HW, HW]
# #                 SA = torch.matmul(Q_Kt, V)
# #                 ### Affine Self attention formulation ###
# #                 import pdb; pdb.set_trace()
# #                 # todo
# #             else:
#                 # import pdb; pdb.set_trace()
#                 Q_Kt = torch.bmm(Q, K.transpose(1,2))
#                 Q_Kt = self.s(Q_Kt / self.scale) # SA_xid : [B, HW, HW]
#                 SA_Y = torch.bmm(Q_Kt,V) # SA_xid : [B, HW, C/2]
#                 # ### Affine Self attention formulation ###
#                 # vec_A = SA @ self.W_a_vec # [HW, 1]
#                 # vec_A = torch.sigmoid(vec_A)
#                 # affout = vec_A * x_tr + SA
#                 ####### 4. Make A and B #######
#                 ### (1) A
#                 A = SA_Y @ self.W_a_mat
#                 ### (2) B
#                 B = SA_Y @ self.W_b_mat
#                 ####### 5. Transform X_tilde #######
#                 x_res = A * x_tilde + B
#             ####### 6. Calculate log det J #######
#             I = torch.ones(A.shape)
#             logA = torch.log(torch.abs(A))
#             log_det_j = torch.sum(logA, dim=2)
#             log_det_j = rearrange(log_det_j, 'b p -> (b p)')
#             if self.isMultiHead:
#                 log_det_j = log_det_j.mean(dim=1)
#             logdet = logdet + log_det_j
#             if self.isMultiHead: ### Concat Multi Head ###
#                 x_res = self._concat(x_res)
#             output = torch.cat((x_res, y_tilde), -1) # [B, HW, C]
#             output = rearrange(output, 'b p c -> (b p) c', p=self.H*self.W)
# #             if self.cfg.use_wandb:
# #                 wandb.log({'logdet': torch.mean(logdet), 'log_det_j': torch.mean(log_det_j),
# #                             'SA':SA.mean(), 'x_tr':x_tr.mean(), 'affout':affout.mean()})
# #         else:
# #             p = self.ch // 2
# #             if self.cfg.action_type == 'toy-example':
# #                 p_size = 1  # options: 1 or 2
# #             else:
# #                 p_size = 14
# #             batch_size = input.shape[0] // (p_size ** 2)
# #             out = rearrange(input, '(b h w) c -> b c h w', h=p_size, w=p_size)
# #             ####### Split ####### H=14, W=14, C=768 / HW=196 <= C/n=384 : n=2,
# #             y_id, y_tr = torch.split(out, self.d_split, dim=-1) # y_identity(self-attention), y_transformation [32, HW196, ch//n]
# #             ####### Self-attention with y_identity #######
# #             qkv = rearrange(x_id, 'b (d p) c -> b c p d', d=1, p=p_size**2) # [B, HW, C/2, 1] add one dim for convolution
# #             q,k,v = self.conv2d_q1(qkv), self.conv2d_k1(qkv), self.conv2d_v1(qkv) # # [B, C/2, HW, 1]
# #             Q = rearrange(q, 'b c p d -> b p (c d)')
# #             K = rearrange(k, 'b c p d -> b p (c d)')
# #             V = rearrange(v, 'b c p d -> b p (c d)') # [b, HW, C/2]
# #             if self.isMultiHead: ### Split to Multi Head ###
# #                 Q, K, V = self._split(Q), self._split(K), self._split(V) # [32, n_head, 196, C/2/n_head]
# #                 SA = Q @ K.transpose(2,3)
# #             else:
# #                 SA = torch.bmm(Q, K.transpose(1,2))
# #             SA = self.s(SA / self.scale)
# #             SA = torch.bmm(SA,V)
# #             ## to do
# #             # attnRank = torch.matrix_rank(attnKQ) # attnKQ : [32, 196, 196] -> rank -> batch num
# #             # attnRank = attnRank.type(torch.FloatTensor).mean()
# #             # id = torch.eye(attnKQ.shape[-1]).cuda() * self.offset # identity matrix for logdet comptation
# #             # logdet_sdp = torch.slogdet(attnKQ+id)[1]
# #             # patch_num = p_size ** 2
# #             # for i in range(batch_size):
# #             #     logdet[i*patch_num:(i+1)*patch_num] = logdet[i*patch_num:(i+1)*patch_num] - logdet_sdp[i]
# #             # # logdet_sdp = torch.slogdet(attn+id)[1]*p*(p//2)*self.ch
# #             # # logdet = logdet - logdet_sdp
# #             # attn_inv = torch.inverse(attnKQ+id)
# #             # out_attn = torch.matmul(attn_inv, V)
# #             # if self.isMultiHead: ### Concat Multi Head ###
# #             #     out_attn = self._concat(out_attn)
# #             # out_attn = rearrange(out_attn, 'b (h w) c -> b c h w',h=p_size, w=p_size) #[32, dim, 14, 14]
# #             # output = torch.cat((qk, out_attn), 1) #[32, 786, 14, 14]
# #             # output = rearrange(output, 'b c h w -> (b h w) c', h=p_size, w=p_size)
# #         return output, logdet
