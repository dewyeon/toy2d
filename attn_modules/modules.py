import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ParallelPermute(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super(ParallelPermute, self).__init__()
        # print('dims in', dims_in)
        # exit()
        self.n_inputs = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]

        np.random.seed(seed)
        perm, perm_inv = self.get_random_perm(0)
        self.perm = [perm]
        self.perm_inv = [perm_inv]

        for i in range(1, self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, i):
        perm = np.random.permutation(self.in_channels[i])
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i

        perm = torch.LongTensor(perm)
        perm_inv = torch.LongTensor(perm_inv)
        return perm, perm_inv

    def forward(self, x, rev=False):
        # import pdb; pdb.set_trace()
        x = rearrange(x, '(B H W) C -> B C H W', H=1 ,W=1)
        if not rev:
            for i in range(self.n_inputs):
                x = x[:, self.perm[i]]
        else:
            for i in range(self.n_inputs):
                x = x[:, self.perm_inv[i]]
        x = rearrange(x, 'B C H W -> (B H W) C')
        return x

    def jacobian(self, x, rev=False):
        # TODO: use batch size, set as nn.Parameter so cuda() works
        return [0.] * self.n_inputs

    def output_dims(self, input_dims):
        return input_dims

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet