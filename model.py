import math
import torch
from torch import nn
from custom_models import *
from custom_models.my_all_in_one_block import MyAllInOneBlock
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
from attn_modules.attention_step import *
from attn_modules.attnBlock import *
from attn_modules.magicAttnBlock import *
from attn_modules.invertAttnBlock import *
from attn_modules.res_invertAttnBlock import *
from attn_modules.only_attnBlock import *

# Swish activation function
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.beta = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x * self.sigmoid(self.beta * x) / 1.1

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))

def deep_subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def freia_flow_head(c, n_feat):
    coder = Ff.SequenceINN(n_feat)
    print('NF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_cflow_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.coupling_blocks):
        if c.is_attention:
            coder.append(isdpAttnBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                global_affine_type='SOFTPLUS', permute_soft=True)
        else:
            coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def freia_invertible_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.K_steps):
        if c.is_attention:
            coder.append(MagicAttnBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                    global_affine_type='SOFTPLUS', permute_soft=True, split_dim=c.split_dim, n_head=c.n_head, is_multiHead=c.multihead_isdp, cfg=c)
            # coder.append(isdpAttnBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                # global_affine_type='SOFTPLUS', permute_soft=True, n_head=c.n_head, is_multiHead=c.multihead_isdp)
        else:
            coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                global_affine_type='SOFTPLUS', permute_soft=True)
        # coder.append(MyAllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
        #   global_affine_type='SOFTPLUS', permute_soft=True)
        # To do: NICECouplingBlock, RNVPCouplingBlock, GLOWCouplingBlock
    return coder

def invertible_attn_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.K_steps):
        coder.append(InvAttnBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                    global_affine_type='SOFTPLUS', permute_soft=True, split_dim=c.split_dim, n_head=c.n_head, is_multiHead=c.multihead_isdp, cfg=c)
        # coder.append(MyAllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
        #   global_affine_type='SOFTPLUS', permute_soft=True)
        # To do: NICECouplingBlock, RNVPCouplingBlock, GLOWCouplingBlock
    return coder

def invertible_resolution_attn_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.K_steps):
        coder.append(ResInvAttnBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
                    global_affine_type='SOFTPLUS', permute_soft=True, split_dim=c.split_dim, n_head=c.n_head, is_multiHead=c.multihead_isdp, cfg=c)
        # coder.append(MyAllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
        #   global_affine_type='SOFTPLUS', permute_soft=True)
        # To do: NICECouplingBlock, RNVPCouplingBlock, GLOWCouplingBlock
    return coder

def only_invertible_attention(c, n_feat):
    # n_cond = c.condition_vec
    n_cond = 16
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.K_steps):
        coder.append(Only_isdpAttnBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)

    return coder

def load_decoder_arch(c, dim_in):
    if c.dec_arch == 'freia-flow':
        decoder = freia_flow_head(c, dim_in)
    elif c.dec_arch == 'freia-cflow':
        decoder = freia_cflow_head(c, dim_in)
    elif c.dec_arch == 'freia-invertible':
        decoder = freia_invertible_head(c, dim_in)
    elif c.dec_arch == 'invertible-attention':
        decoder = invertible_attn_head(c, dim_in)
    elif c.dec_arch == 'invertible-attention-resolution':
        decoder = invertible_resolution_attn_head(c, dim_in)
    elif c.dec_arch == "only_invertible_attention":
        dim_in = 16
        # c.condition_vec = 16
        decoder = only_invertible_attention(c, dim_in)    
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.dec_arch))
    #print(decoder)
    return decoder


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


"""
using our framework
"""
def load_encoder_arch_for_invertible(c, L):
    hook_layers = ['layer'+str(i) for i in range(L)]
    hook_dims = list()
    hook_cnt = 0

    if 'vit' in c.enc_arch:
        vit_list = ['vit_base_patch16_224', 'vit_tiny_patch16_224', 'vit_small_patch32_224', 'vit_small_patch16_224','vit_base_patch32_224','vit_large_patch32_224','vit_large_patch16_224','vit_tiny_patch16_384','vit_small_patch32_384','vit_small_patch16_384','vit_base_patch32_384','vit_base_patch16_384','vit_base_patch8_224','vit_large_patch32_384','vit_large_patch16_384']

        if c.enc_arch in vit_list:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

        if c.enc_layer == '3layer':
            if L >= 3:
                encoder.blocks[10].register_forward_hook(get_activation(hook_layers[hook_cnt]))
                hook_dims.append(encoder.blocks[6].mlp.fc2.out_features)
                hook_cnt = hook_cnt + 1
            if L >= 2:
                encoder.blocks[2].register_forward_hook(get_activation(hook_layers[hook_cnt]))
                hook_dims.append(encoder.blocks[6].mlp.fc2.out_features)
                hook_cnt = hook_cnt + 1
            if L >= 1:
                encoder.blocks[6].register_forward_hook(get_activation(hook_layers[hook_cnt]))
                hook_dims.append(encoder.blocks[6].mlp.fc2.out_features)
                hook_cnt = hook_cnt + 1

        elif c.enc_layer == '1layer':
            # toy setting...
            encoder.blocks[-1].register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[-1].mlp.fc2.out_features)
            hook_cnt = hook_cnt + 1

        elif c.enc_layer == 'grid_layer':
            # toy setting...
            encoder.blocks[c.grid_layer].register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[c.grid_layer].mlp.fc2.out_features)
            hook_cnt = hook_cnt + 1
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
    
    elif 'cait' in c.enc_arch:
        cait_list = ['cait_xxs24_224', 'cait_xxs24_384', 'cait_xxs36_224', 'cait_xxs36_384', 'cait_xs24_384', 'cait_s24_224', 'cait_s24_384', 'cait_s36_384', 'cait_m36_384']
        if  c.enc_arch in cait_list:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

        if L == 1:
            encoder.blocks[c.grid_layer].register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            hook_cnt = hook_cnt + 1
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

    elif 'deit' in c.enc_arch:
        if c.enc_arch == 'deit_tiny_patch16_224':
            encoder = timm.create_model('deit_tiny_patch16_224', pretrained=True)
        elif c.enc_arch == 'deit_small_patch16_224':
            encoder = timm.create_model('deit_small_patch16_224', pretrained=True)
        elif c.enc_arch == 'deit_base_patch16_224':
            encoder = timm.create_model('deit_base_patch16_224', pretrained=True)
        elif c.enc_arch == 'deit_small_patch16_224':
            encoder = timm.create_model('deit_small_patch16_224', pretrained=True)
        elif c.enc_arch == 'deit_tiny_distilled_patch16_224':
            encoder = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True)
        elif c.enc_arch == 'deit_small_distilled_patch16_224':
            encoder = timm.create_model('deit_small_distilled_patch16_224', pretrained=True)
        elif c.enc_arch == 'deit_base_distilled_patch16_224':
            encoder = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

        if L == 1:
            encoder.blocks[c.grid_layer].register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            hook_cnt = hook_cnt + 1
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

    elif 'swin' in c.enc_arch:
        swin_list = ['swin_base_patch4_window12_384', 'swin_base_patch4_window7_224', 'swin_large_patch4_window12_384', 
                    'swin_large_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_tiny_patch4_window7_224',
                    'swin_base_patch4_window12_384_in22k','swin_base_patch4_window7_224_in22k',
                    'swin_large_patch4_window12_384_in22k','swin_large_patch4_window7_224_in22k']
        if  c.enc_arch in swin_list:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
            if L == 1:
                if c.enc_arch == 'swin_tiny_patch4_window7_224' and c.grid_layer >= 6:
                    raise NotImplementedError('{} is only 6 blocks!'.format(c.enc_arch))
                else:
                    if c.enc_arch == 'swin_tiny_patch4_window7_224':
                        encoder.layers[2].blocks[c.grid_layer].register_forward_hook(get_activation(hook_layers[hook_cnt]))
                        hook_dims.append(encoder.layers[2].blocks[c.grid_layer].mlp.fc2.out_features)
                    else:
                        encoder.layers[2].blocks[c.grid_layer].register_forward_hook(get_activation(hook_layers[hook_cnt]))
                        hook_dims.append(encoder.layers[2].blocks[c.grid_layer].mlp.fc2.out_features)
                        hook_cnt = hook_cnt + 1
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

    elif 'resnet' in c.enc_arch:
        if   c.enc_arch == 'resnet18':
            encoder = resnet18(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet34':
            encoder = resnet34(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet50':
            encoder = resnet50(pretrained=True, progress=True)
        elif c.enc_arch == 'resnext50_32x4d':
            encoder = resnext50_32x4d(pretrained=True, progress=True)
        elif c.enc_arch == 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

        if L >= 3:
            encoder.layer2.register_forward_hook(get_activation(hook_layers[hook_cnt]))
            if 'wide' in c.enc_arch:
                hook_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                hook_dims.append(encoder.layer2[-1].conv2.out_channels)
            hook_cnt = hook_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(hook_layers[hook_cnt]))
            if 'wide' in c.enc_arch:
                hook_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                hook_dims.append(encoder.layer3[-1].conv2.out_channels)
            hook_cnt = hook_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(hook_layers[hook_cnt]))
            if 'wide' in c.enc_arch:
                hook_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                hook_dims.append(encoder.layer4[-1].conv2.out_channels)
            hook_cnt = hook_cnt + 1

    elif 'efficient' in c.enc_arch:
        if 'b5' in c.enc_arch:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
            blocks = [-2, -3, -5]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[blocks[2]][-1].bn3.register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[blocks[2]][-1].bn3.num_features)
            hook_cnt = hook_cnt + 1
        if L >= 2:
            encoder.blocks[blocks[1]][-1].bn3.register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[blocks[1]][-1].bn3.num_features)
            hook_cnt = hook_cnt + 1
        if L >= 1:
            encoder.blocks[blocks[0]][-1].bn3.register_forward_hook(get_activation(hook_layers[hook_cnt]))
            hook_dims.append(encoder.blocks[blocks[0]][-1].bn3.num_features)
            hook_cnt = hook_cnt + 1

    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))

    return encoder, hook_layers, hook_dims
        

def load_encoder_arch(c, L): 
    # encoder pretrained on natural images:
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer'+str(i) for i in range(L)]
    if 'resnet' in c.enc_arch:
        if   c.enc_arch == 'resnet18':
            encoder = resnet18(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet34':
            encoder = resnet34(pretrained=True, progress=True)
        elif c.enc_arch == 'resnet50':
            encoder = resnet50(pretrained=True, progress=True)
        elif c.enc_arch == 'resnext50_32x4d':
            encoder = resnext50_32x4d(pretrained=True, progress=True)
        elif c.enc_arch == 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=True, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
    elif 'vit' in c.enc_arch:
        if  c.enc_arch == 'vit_base_patch16_224':
            encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        elif  c.enc_arch == 'vit_base_patch16_384':
            encoder = timm.create_model('vit_base_patch16_384', pretrained=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[10].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[2].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[6].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[6].mlp.fc2.out_features)
            pool_cnt = pool_cnt + 1
    elif 'efficient' in c.enc_arch:
        if 'b5' in c.enc_arch:
            encoder = timm.create_model(c.enc_arch, pretrained=True)
            blocks = [-2, -3, -5]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder.blocks[blocks[2]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[2]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder.blocks[blocks[1]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[1]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder.blocks[blocks[0]][-1].bn3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder.blocks[blocks[0]][-1].bn3.num_features)
            pool_cnt = pool_cnt + 1
    elif 'mobile' in c.enc_arch:
        if  c.enc_arch == 'mobilenet_v3_small':
            encoder = mobilenet_v3_small(pretrained=True, progress=True).features
            blocks = [-2, -5, -10]
        elif  c.enc_arch == 'mobilenet_v3_large':
            encoder = mobilenet_v3_large(pretrained=True, progress=True).features
            blocks = [-2, -5, -11]
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        #
        if L >= 3:
            encoder[blocks[2]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[2]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 2:
            encoder[blocks[1]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[1]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
        if L >= 1:
            encoder[blocks[0]].block[-1][-3].register_forward_hook(get_activation(pool_layers[pool_cnt]))
            pool_dims.append(encoder[blocks[0]].block[-1][-3].out_channels)
            pool_cnt = pool_cnt + 1
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
    #
    return encoder, pool_layers, pool_dims
