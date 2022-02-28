import torch
import os
import numpy as np
import math
from einops import rearrange
import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))

def get_logp_z(z):
    # import pdb; pdb.set_trace()
    C = 2
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1)
    # logp = - C * 0.5 * math.log(math.pi * 2) - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

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

@torch.no_grad()
def plot(batch_id, model, potential_or_sampling_fn, args):
    model.eval()
    
    n_pts = args.plot_resolution
    range_lim = 4

    # construct test points
    test_grid = setup_grid(range_lim, n_pts, args)

    # plot
    if args.density_matching:
        if args.flow == "boosted":
            #plt_height = max(1, int(np.ceil(np.sqrt(args.num_components + 2))))
            #plt_width = max(2, int(np.ceil((args.num_components + 2) / plt_height)))
            plt_width = 2
            plt_height = 1
            fig, axs = plt.subplots(plt_height, plt_width, figsize=(12,12), subplot_kw={'aspect': 'equal'}, squeeze=False)
            plot_potential(potential_or_sampling_fn, axs[0, 0], test_grid, n_pts)
            plot_flow_samples(model, axs[0, 1], n_pts, args.batch_size, args)
            #plot_boosted_inv_flow_density(model, axs, test_grid, n_pts, args.batch_size, args, plt_height, plt_width)
        else:
            fig, axs = plt.subplots(1, 3, figsize=(16,8), subplot_kw={'aspect': 'equal'}, squeeze=False)
            plot_potential(potential_or_sampling_fn, axs[0, 0], test_grid, n_pts)
            plot_flow_samples(model, axs[0, 1], n_pts, args.batch_size, args)
            plot_inv_flow_density(model, axs[0, 2], test_grid, n_pts, args.batch_size, args)
            
    else:
        if args.flow == "boosted":
            plt_width =  max(2, int(np.ceil(np.sqrt(args.num_components))))
            plt_height = max(2, int(np.ceil(np.sqrt(args.num_components))) + 1)
            #plt_height = max(1, int(np.ceil(np.sqrt(args.num_components + 2))))
            #plt_width = max(1, int(np.ceil((args.num_components + 2) / plt_height)))
            fig, axs = plt.subplots(plt_height, plt_width, figsize=(12,12), subplot_kw={'aspect': 'equal'}, squeeze=False)
            plot_samples(potential_or_sampling_fn, axs[0,0], range_lim, n_pts)
            total_prob = plot_boosted_fwd_flow_density(model, axs, test_grid, n_pts, args.batch_size, args)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12,12), subplot_kw={'aspect': 'equal'})
            plot_samples(potential_or_sampling_fn, axs[0], range_lim, n_pts)
            plot_fwd_flow_density(model, axs[1], test_grid, n_pts, args.batch_size, args)

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    #plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.tight_layout()

    title = f'{args.dataset.title()}: {args.flow.title()} Flow, K={args.K_steps}'
    title += f', C={args.num_components}' if args.flow == "boosted" else ''
    title += f', Reg={args.regularization_rate:.2f}' if args.flow == "boosted" and args.density_matching else ''
    annealing_type = f', Annealed' if args.min_beta < 1.0 else ', No Annealing'
    title += annealing_type if args.density_matching else ''  # annealing isn't done for density sampling
    fig.suptitle(title, y=0.98, fontsize=20)
    fig.subplots_adjust(top=0.85)

    # save
    fname = f'{args.dataset}_{args.flow}_K{args.K_steps}_bs{args.batch_size}'
    fname += f'_C{args.num_components}_reg{int(100*args.regularization_rate):d}_{args.component_type}' if args.flow == 'boosted' else ''
    fname += f'_{args.coupling_network}{args.coupling_network_depth}_hsize{args.h_size}' if args.flow == 'realnvp' else ''
    fname += f'_hidden{args.coupling_network_depth}_hsize{args.h_size}' if args.flow == 'iaf' else ''
    fname += '_annealed' if args.min_beta < 1.0 else ''
    fname += '_lr_scheduling' if not args.no_lr_schedule else ''
    plt.savefig(os.path.join(args.snap_dir, fname + f'_step{batch_id:07d}.png'))
    plt.close()


    # plot densities using gaussian interpolation
    if args.density_matching:
        if args.flow == "boosted":
            plot_boosted_inv_flow(model, batch_id, 1000, args.batch_size, args)
        else:
            plot_inv_flow(model, batch_id, 1000, args.batch_size, args)
    

    # PLOT THE FINAL RESULT IF THIS IS THE LAST BATCH
    if batch_id == args.num_steps:
        fig, axs = plt.subplots(1, 2, figsize=(12,12), subplot_kw={'aspect': 'equal'})
        if args.density_matching:
            plot_potential(potential_or_sampling_fn, axs[0], test_grid, n_pts)
            plot_flow_samples(model, axs[1], n_pts, args.batch_size, args)    
        else:
            plot_samples(potential_or_sampling_fn, axs[0], range_lim, n_pts)
            if args.flow == "boosted":
                xx, yy, zz = test_grid
                axs[1].pcolormesh(xx, yy, total_prob, cmap=plt.cm.viridis)
                axs[1].set_facecolor(plt.cm.viridis(0.))
                axs[1].set_title('Boosted Density - All Components', fontdict={'fontsize': 20})
            else:
                plot_fwd_flow_density(model, axs[1], test_grid, n_pts, args.batch_size, args)
                
        for ax in plt.gcf().axes: format_ax(ax, range_lim)
        #plt.tight_layout(rect=[0, 0, 1.0, 0.95])
        plt.tight_layout()
        title = f'{args.dataset.title()}: {args.flow.title()} Flow, K={args.num_flows}'
        title += f', Annealed' if args.min_beta < 1.0 else ', No Annealing'
        title += f', C={args.num_components}, Reg={args.regularization_rate:.2f}' if args.flow == "boosted" else ''
        fig.suptitle(title, y=0.98, fontsize=20)
        fig.subplots_adjust(top=0.85)   # too much?

        plt.savefig(os.path.join(args.snap_dir, fname + '.png'))
        plt.close()


def setup_grid(range_lim, n_pts, args):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(args.device)


def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

    
def plot_potential(potential_fn, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    ax.pcolormesh(xx, yy, torch.exp(-1.0 * potential_fn(zz)).view(n_pts, n_pts).cpu().data, cmap=plt.cm.viridis)
    ax.set_title('Target Density', fontdict={'fontsize': 20})
    
    
def plot_samples(samples_fn, ax, range_lim, n_pts):
    samples = samples_fn(n_pts**2).numpy()
    ax.hist2d(samples[:,0], samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.viridis)
    ax.set_title('Target Samples', fontdict={'fontsize': 20})

    
def plot_flow_samples(model, ax, n_pts, batch_size, args):
    z = model.base_dist.sample((n_pts**2,))
    if args.flow == "boosted":
        caption = f" - All Components"
        zk = torch.cat([model.flow(z_, sample_from="1:c")[0][-1] for z_ in z.split(batch_size, dim=0)], 0)
    else:
        caption = f""
        zk = torch.cat([model.flow(z_)[0] for z_ in z.split(batch_size, dim=0)], 0)

    zk = torch.clamp(zk, min=-25.0, max=25.0)
    zk = zk.cpu().numpy()
    
    # plot
    ax.hist2d(zk[:,0], zk[:,1], bins=n_pts, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.))
    ax.set_title('Flow Samples' + caption, fontdict={'fontsize': 20})

    
def plot_fwd_flow_density(model, ax, test_grid, n_pts, batch_size, args):
    """
    plots square grid and flow density; where density under the flow is exp(log_flow_base_dist + logdet)
    """
    xx, yy, zz = test_grid
    
    # compute posterior approx density
    zzk, logdet = [], []
    B = batch_size
    H=1; W=1
    P = args.condition_vec
    pos = positionalencoding2d(P, H, W)
    cond_list = []
    for layer in range(args.L_layers):
        res = torch.zeros(args.L_layers, H, W)
        res[layer] = 1
        cond = torch.cat((pos, res), dim=0).to(args.device).unsqueeze(0).repeat(B // args.L_layers, 1, 1, 1)
        cond_list.append(cond)
    #### L=2일때 수정해야함!!!!!!!!
    c_r = rearrange(cond, 'b c h w -> (b h w) c')

    for zz_i in zz.split(batch_size, dim=0):    
        #zzk_i, logdet_i = model.flow(zz_i)
        zzk_i, logdet_i = model(zz_i, [c_r,])
        zzk += [zzk_i]
        logdet += [logdet_i]
        
    zzk, logdet = torch.cat(zzk, 0), torch.cat(logdet, 0)
    q_log_prob = get_logp_z(zzk) / 2
    log_prob = q_log_prob + logdet
    prob = log_prob.exp().cpu()

    # plot
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts).data, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.))
    ax.set_title('Flow Density', fontdict={'fontsize': 20})


def plot_boosted_fwd_flow_density(model, axs, test_grid, n_pts, batch_size, args, batch_id=None):
    """
    plots square grid and flow density; where density under the flow is exp(log_flow_base_dist + logdet)
    """
    xx, yy, zz = test_grid
    num_fixed_plots = 2  # every image will show the true samples and the density for the full model
    #plt_height = max(1, int(np.ceil(np.sqrt(args.num_components + num_fixed_plots))))
    #plt_width = max(1, int(np.ceil((args.num_components + num_fixed_plots) / plt_height)))

    plt_width =  max(2, int(np.ceil(np.sqrt(args.num_components))))
    plt_height = max(2, int(np.ceil(np.sqrt(args.num_components))) + 1)
            
    total_prob = torch.zeros(n_pts, n_pts)
    num_components_to_plot = max(1, args.num_components if model.all_trained else model.component + 1)
    for c in range(num_components_to_plot):
        if model.rho[c] == 0.0:
            continue
        
        #row = int(np.floor((c + num_fixed_plots) / plt_width))
        #col = int((c + num_fixed_plots) % plt_width)
        row = int(1 + np.floor(c / plt_width))
        col = int(c % plt_width)

        zzk, logdet = [], []
        for zz_i in zz.split(batch_size, dim=0):
            ZZ_i, _, _, logdet_i, _ = model(x=zz_i, components=c)
            zzk += [ZZ_i]    
            logdet += [logdet_i]

        zzk, logdet = torch.cat(zzk, 0), torch.cat(logdet, 0)
        q_log_prob = model.base_dist.log_prob(zzk).sum(1)
        log_prob = q_log_prob + logdet
        prob = log_prob.exp().cpu().view(n_pts,n_pts).data

        # plot component c
        axs[row,col].pcolormesh(xx, yy, prob, cmap=plt.cm.viridis)
        axs[row,col].set_facecolor(plt.cm.viridis(0.))
        axs[row,col].set_title(f'c={c}', fontdict={'fontsize': 20})

        # save total model probs
        total_prob += log_prob.cpu().view(n_pts, n_pts).data * model.rho[c]

    # plot full model
    total_prob = torch.exp(total_prob / torch.sum(model.rho[0:num_components_to_plot]))
    axs[0,1].pcolormesh(xx, yy, total_prob, cmap=plt.cm.viridis)
    axs[0,1].set_facecolor(plt.cm.viridis(0.))
    axs[0,1].set_title('GBF - All Components', fontdict={'fontsize': 20})
    return total_prob

    
def plot_inv_flow_density(model, ax, test_grid, n_pts, batch_size, args):
    """
    plots transformed grid and density; where density is exp(loq_flow_base_dist - logdet)
    """        
    xx, yy, zz = test_grid
    
    # compute posterior approx density
    zzk, logdet = [], []
    for zz_i in zz.split(batch_size, dim=0):
        zzk_i, logdet_i = model.flow(zz_i)
        zzk += [zzk_i]
        logdet += [logdet_i]
    
    zzk, logdet = torch.cat(zzk, 0), torch.cat(logdet, 0)
    log_q0 = model.base_dist.log_prob(zz).sum(1)
    log_qk = log_q0 - logdet
    qk = log_qk.exp().cpu()
    zzk = zzk.cpu()
    
    # plot
    ax.pcolormesh(zzk[:,0].view(n_pts,n_pts).data, zzk[:,1].view(n_pts,n_pts).data, qk.view(n_pts,n_pts).data, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.0))
    ax.set_title('Flow Density', fontdict={'fontsize': 20})

    
def plot_inv_flow(model, batch_id, n_pts, batch_size, args):
    fname = f'{args.dataset}_{args.flow}_K{args.num_flows}_bs{args.batch_size}'
    fname += f'_{args.coupling_network}{args.coupling_network_depth}_hsize{args.h_size}' if args.component_type == 'realnvp' or args.flow == 'realnvp' else ''
    fname += f'_hidden{args.coupling_network_depth}_hsize{args.h_size}' if args.flow == 'iaf' else ''
    fname += '_annealed' if args.min_beta < 1.0 else ''
    fname += '_lr_scheduling' if not args.no_lr_schedule else ''

    Z = np.hstack([model.flow(torch.randn(n_pts, 2).to(args.device) * model.base_dist_var + model.base_dist_mean)[0].t().cpu().data.numpy() for _ in range(n_pts)])
        
    H, _, _ = np.histogram2d(Z[0], Z[1], bins=(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05)))
    plt.figure(figsize=(12, 12))
    plt.imshow(H.T, interpolation='gaussian')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(os.path.join(args.snap_dir, f'final_{fname}_step{batch_id:07d}.png'))


def plot_boosted_inv_flow_density(model, axs, test_grid, n_pts, batch_size, args, plt_height, plt_width):
    """
    plots transformed grid and density; where density is exp(loq_flow_base_dist - logdet)
    """
    xx, yy, zz = test_grid
    num_fixed_plots = 2  # every image will show the true density and samples from the full model

    num_components_to_plot = args.num_components if model.all_trained else model.component + 1
    for c in range(num_components_to_plot):
        if model.rho[c] == 0.0:
            continue
        
        row = int(np.floor((c + num_fixed_plots) / plt_width))
        col = int((c + num_fixed_plots) % plt_width)

        zzk, logdet = [], []
        for zz_i in zz.split(batch_size, dim=0):
            ZZ_i, logdet_i = model.component_forward_flow(zz_i, c)
            zzk += [ZZ_i[-1]]  # grab K-th element
            logdet += [logdet_i]

        zzk, logdet = torch.cat(zzk, 0), torch.cat(logdet, 0)
        log_q0 = model.base_dist.log_prob(zz).sum(1)
        log_qk = log_q0 - logdet
        qk = log_qk.exp().cpu()
        zzk = zzk.cpu()

        # plot component c
        axs[row,col].pcolormesh(zzk[:,0].view(n_pts,n_pts).data, zzk[:,1].view(n_pts,n_pts).data, qk.view(n_pts,n_pts).data,
                                cmap=plt.cm.viridis)
        axs[row,col].set_facecolor(plt.cm.viridis(0.0))
        axs[row,col].set_title(f'Boosted Flow Density for c={c}', fontdict={'fontsize': 20})


def plot_boosted_inv_flow(model, batch_id, n_pts, batch_size, args):
    """
    plots transformed grid and density; where density is a gaussian interpolation of the model's samples
    """
    fname = f'{args.dataset}_{args.flow}_K{args.num_flows}_bs{args.batch_size}'
    fname += f'_C{args.num_components}_reg{int(100*args.regularization_rate):d}_{args.component_type}'
    fname += f'_{args.coupling_network}{args.coupling_network_depth}_hsize{args.h_size}' if args.component_type == 'realnvp' or args.flow == 'realnvp' else ''
    fname += '_annealed' if args.min_beta < 1.0 else ''
    fname += '_lr_scheduling' if not args.no_lr_schedule else ''

    Z = []
    num_components_to_plot = args.num_components if model.all_trained else model.component + 1
    for c in range(num_components_to_plot):        
        zc = np.hstack([model.component_forward_flow(
            torch.randn(n_pts, 2).to(args.device) * model.base_dist_var + model.base_dist_mean, c)[0][-1].t().cpu().data.numpy() for _ in range(n_pts)])
        
        num_sampled = int(np.ceil(( model.rho[c] / model.rho.sum() ) * n_pts * n_pts))
        Z.append(zc[:, 0:num_sampled])

        # plot component c
        Hc, _, _ = np.histogram2d(zc[0], zc[1], bins=(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05)))
        plt.figure(figsize=(12, 12))
        plt.imshow(Hc.T, interpolation='gaussian')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(os.path.join(args.snap_dir, f'{c}_{fname}_step{batch_id:07d}.png'))

        if model.component == 0 and not model.all_trained:
            # don't bother plotting components that haven't been trained at all
            break


    # plot full model
    Z = np.hstack(Z)
    H, _, _ = np.histogram2d(Z[0], Z[1], bins=(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05)))
    plt.figure(figsize=(12, 12))
    plt.imshow(H.T, interpolation='gaussian')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(os.path.join(args.snap_dir, f'final_{fname}_step{batch_id:07d}.png'))


def plot_q0_density(model, ax, test_grid, n_pts, batch_size, args):
    """
    Plot the base distribution (some type of standard gaussian)
    """        
    xx, yy, zz = test_grid
    log_q0 = model.base_dist.log_prob(zz).sum(1)    
    q0 = log_q0.exp().cpu()
    
    # plot
    ax.pcolormesh(zz[:,0].view(n_pts,n_pts).data, zz[:,1].view(n_pts,n_pts).data, q0.view(n_pts,n_pts).data, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.))
    ax.set_title('Base q_0 Density', fontdict={'fontsize': 20})
    
