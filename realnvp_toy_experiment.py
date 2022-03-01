import argparse
import datetime
import torch
import numpy as np
import math
import random
import os
import logging
import torch.optim as optim
# from tensorboardX import SummaryWriter
from collections import Counter

from models.boosted_vae import BoostedVAE
from models.boosted_flow import BoostedFlow
from models.realnvp import RealNVPFlow
from models.iaf import IAFFlow
from models.planar import PlanarFlow
from models.radial import RadialFlow
from models.liniaf import LinIAFFlow
from models.affine import AffineFlow
from models.nlsq import NLSqFlow

from utils.realnvp_density_plotting import plot
from utils.load_data import make_toy_density, make_toy_sampler
from utils.utilities import init_log, softmax
from optimization.optimizers import GradualWarmupScheduler
from utils.distributions import log_normal_diag, log_normal_standard, log_normal_normalized


logger = logging.getLogger(__name__)


TOY_DATASETS = ["8gaussians", "2gaussians", "1gaussian",  "swissroll", "rings", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "circles", "joint_gaussian"]
ENERGY_FUNS = ['u0', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6']
G_MAX_LOSS = -10.0

parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')
parser.add_argument('--experiment_name', type=str, default="toy",
                    help="A name to help identify the experiment being run when training this model.")
parser.add_argument('--dataset', type=str, default='8gaussians', help='Dataset choice.', choices=TOY_DATASETS + ENERGY_FUNS)
parser.add_argument('--mog_sigma', type=float, default=1.5, help='Variance in location of mixture of gaussian data.',
                    choices=[i / 100.0 for i in range(50, 250)])
parser.add_argument('--mog_clusters', type=int, default=6, help='Number of clusters to use in the mixture of gaussian data.',
                    choices=range(1,13))

# seeds
parser.add_argument('--manual_seed', type=int, default=123,
                    help='manual seed, if not given resorts to random seed.')

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
                    help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

# Reporting
parser.add_argument('--log_interval', type=int, default=1000,
                    help='how many batches to wait before logging training status. Set to <0 to turn off.')
parser.add_argument('--plot_interval', type=int, default=1000,
                    help='how many batches to wait before creating reconstruction plots. Set to <0 to turn off.')
parser.add_argument('--no_tensorboard', dest="tensorboard", action="store_false", help='Turns off saving results to tensorboard.')
parser.set_defaults(tensorboard=True)

parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/toy_experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--print_log', dest="print_log", action="store_true", help='Add this flag to have progress printed to log (rather just than saved to a file).')
parser.set_defaults(print_log=False)

sr = parser.add_mutually_exclusive_group(required=False)
sr.add_argument('--save_results', action='store_true', dest='save_results', help='Save results from experiments.')
sr.add_argument('--discard_results', action='store_false', dest='save_results', help='Do NOT save results from experiments.')
parser.set_defaults(save_results=True)
parser.add_argument('--plot_resolution', type=int, default=250, help='how many points to plot, higher gives better resolution')

# optimization settings
parser.add_argument('--num_steps', type=int, default=100000, help='number of training steps to take (default: 100000)')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--regularization_rate', type=float, default=0.8, help='Regularization penalty for boosting.')
parser.add_argument('--iters_per_component', type=int, default=10000, help='how often to train each boosted component before changing')
parser.add_argument('--max_beta', type=float, default=1.0, help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')
parser.add_argument('--no_annealing', action='store_true', default=False, help='disables annealing while training')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter in Adamax')
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')
parser.add_argument('--lr_schedule', type=str, default=None, help="Type of LR schedule to use.", choices=['plateau', 'cosine', None])
parser.add_argument('--lr_restarts', type=int, default=1, help='If using a cyclic/cosine learning rate, how many times should the LR schedule restart? Must evenly divide epochs')
parser.add_argument('--patience', type=int, default=5000, help='If using LR schedule, number of steps before reducing LR.')
parser.add_argument("--max_grad_clip", type=float, default=0, help="Max gradient value (clip above max_grad_clip, 0 for off)")
parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Max norm of gradient (clip above max_grad_norm, 0 for off)")
parser.add_argument("--warmup_iters", type=int, default=0, help="Use this number of iterations to warmup learning rate linearly from zero to learning rate")

# flow parameters
parser.add_argument('--flow', type=str, default='realnvp',
                    choices=['planar', 'radial', 'liniaf', 'affine', 'nlsq', 'boosted', 'iaf', 'realnvp'],
                    help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('--num_flows', type=int, default=2, help='Number of flow layers, ignored in absence of flows')

parser.add_argument('--h_size', type=int, default=64, help='Width of layers in base networks of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--coupling_network_depth', type=int, default=1, help='Number of extra hidden layers in the base network of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--coupling_network', type=str, default='tanh', choices=['relu', 'residual', 'tanh', 'random', 'mixed'],
                    help='Base network for RealNVP coupling layers. Random chooses between either Tanh or ReLU for every network, whereas mixed uses ReLU for the T network and TanH for the S network.')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='Disables batch norm in realnvp layers')
parser.set_defaults(batch_norm=True)
parser.add_argument('--z_size', type=int, default=2, help='Size of base distibution, should be the same as data input size.')

# Boosting parameters
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'],
                    help='Initialization scheme for boosted parameter rho')
parser.add_argument('--rho_iters', type=int, default=100, help='Maximum number of SGD iterations for training boosting weights')
parser.add_argument('--rho_lr', type=float, default=0.005, help='Initial learning rate used for training boosting weights')
parser.add_argument('--num_components', type=int, default=4,
                    help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='affine', choices=['liniaf', 'affine', 'nlsq', 'realnvp'],
                    help='When flow is boosted -- what type of flow should each component implement.')



def parse_args(main_args=None):
    """
    Parse command line arguments and compute number of cores to use
    """
    args = parser.parse_args(main_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.density_matching = args.dataset.startswith('u')
    args.dynamic_binarization = False
    args.input_type = 'binary'
    args.input_size = [2]
    args.density_evaluation = True
    args.shuffle = True
    args.train_size = args.iters_per_component
    args.learn_top = False
    args.y_classes = None
    args.y_condition = None
    args.sample_size = args.z_size

    # Set a random seed if not given one
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    # intialize snapshots directory for saving models and results
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    args.experiment_name = args.experiment_name + "_" if args.experiment_name is not None else ""
    args.snap_dir = os.path.join(args.out_dir, args.experiment_name + args.flow)

    lr_schedule = f'_lr{str(args.learning_rate)[2:]}'
    if args.lr_schedule is None or args.no_lr_schedule:
        args.no_lr_schedule = True
        args.lr_schedule = None
    else:
        args.no_lr_schedule = False
        lr_schedule += f'{args.lr_schedule}'

    if args.dataset in ['u5', 'mog']:
        dataset = f"{args.dataset}_s{int(100 * args.mog_sigma)}_c{args.mog_clusters}"
    else:
        dataset = args.dataset

    args.snap_dir += f'_seed{args.manual_seed}' + lr_schedule + '_' + dataset + f"_bs{args.batch_size}"

    args.boosted = args.flow == "boosted"
    if args.flow != 'no_flow':
        args.snap_dir += 'K' + str(args.num_flows)

    if args.flow in ['boosted', 'bagged']:
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization rate should be greater than or equal to zero.")
        args.snap_dir += '_' + args.component_type + '_C' + str(args.num_components)
        args.snap_dir += '_reg' + f'{int(100*args.regularization_rate):d}' if args.density_matching else ''

    if args.flow == 'iaf':
        args.snap_dir += '_hidden' + str(args.coupling_network_depth) + '_hsize' + str(args.h_size)

    if args.flow == "realnvp" or args.component_type == "realnvp":
        args.snap_dir += '_' + args.coupling_network + str(args.coupling_network_depth) + '_hsize' + str(args.h_size)
        
    is_annealed = ""
    if not args.no_annealing and args.min_beta < 1.0:
        is_annealed += "_annealed"
    else:
        args.min_beta = 1.0
        
    args.snap_dir += is_annealed + f'_{args.model_signature}/'
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)

    init_log(args)
    
    # Set up multiple CPU/GPUs
    logger.info("COMPUTATION SETTINGS:")
    logger.info(f"Random Seed: {args.manual_seed}\n")
    if args.cuda:
        logger.info("\tUsing CUDA GPU")
        torch.cuda.set_device(args.gpu_id)
    else:
        logger.info("\tUsing CPU")
        if args.num_workers > 0:
            num_workers = args.num_workers
        else:
            num_workers = max(1, os.cpu_count() - 1)

        logger.info("\tCores available: {} (only requesting {})".format(os.cpu_count(), num_workers))
        torch.set_num_threads(num_workers)
        logger.info("\tConfirmed Number of CPU threads: {}".format(torch.get_num_threads()))

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    return args, kwargs


def init_model(args):
    if args.flow == 'boosted':
        if args.density_matching:
            model = BoostedVAE(args).to(args.device)
        else:
            model = BoostedFlow(args).to(args.device)
    elif args.flow == 'planar':
        model = PlanarFlow(args).to(args.device)
    elif args.flow == 'radial':
        model = RadialFlow(args).to(args.device)
    elif args.flow == 'liniaf':
        model = LinIAFFlow(args).to(args.device)
    elif args.flow == 'affine':
        model = AffineFlow(args).to(args.device)
    elif args.flow == 'nlsq':
        model = NLSqFlow(args).to(args.device)
    elif args.flow == 'iaf':
        model = IAFFlow(args).to(args.device)
    elif args.flow == "realnvp":
        model = RealNVPFlow(args).to(args.device)
    else:
        raise ValueError('Invalid flow choice')

    return model


def init_optimizer(model, args, verbose=True):
    """
    group model parameters to more easily modify learning rates of components (flow parameters)
    """
    #warmup_mult = 1000.0
    #base_lr = (args.learning_rate / warmup_mult) if args.warmup_iters > 0 else args.learning_rate
    if verbose:
        logger.info('OPTIMIZER:')
        logger.info(f"Initializing AdamW optimizer with base learning rate={args.learning_rate}, weight decay={args.weight_decay}.")
    
    if args.flow == 'boosted':
        if verbose:
            logger.info("For boosted model, grouping parameters according to Component Id:")
        
        flow_params = {f"{c}": torch.nn.ParameterList() for c in range(args.num_components)}
        flow_labels = {f"{c}": [] for c in range(args.num_components)}
        vae_params = torch.nn.ParameterList()
        vae_labels = []
        for name, param in model.named_parameters():
            if name.startswith("flow"):
                pos = name.find(".")
                component_id = name[(pos + 1):(pos + 2)]
                flow_params[component_id].append(param)
                flow_labels[component_id].append(name)
            else:
                vae_labels.append(name)
                vae_params.append(param)

        # collect all parameters into a single list
        # the first args.num_components elements in the parameters list correspond boosting parameters
        all_params = []
        for c in range(args.num_components):
            all_params.append(flow_params[f"{c}"])

            if verbose:
                logger.info(f"Grouping [{', '.join(flow_labels[str(c)])}] as Component {c}'s parameters.")

        # vae parameters are at the end of the list (may not exist if doing density estimation)
        if len(vae_params) > 0:
            all_params.append(vae_params)

            if verbose:
                logger.info(f"Grouping [{', '.join(vae_labels)}] as the VAE parameters.\n")
            
        #optimizer = optim.Adamax([{'params': param_group} for param_group in all_params], lr=base_lr, weight_decay=args.weight_decay)
        optimizer = optim.AdamW([{'params': param_group} for param_group in all_params], lr=args.learning_rate, weight_decay=args.weight_decay)
        
    else:
        #optimizer = optim.Adamax(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if verbose:
            logger.info(f"Initializing optimizer for standard models with learning rate={args.learning_rate}.\n")

    if args.no_lr_schedule:
        scheduler = None
    else:
        if args.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=0.5,
                                                                   patience=args.patience,
                                                                   min_lr=1e-5,
                                                                   verbose=True,
                                                                   threshold_mode='abs')
            if verbose:
                logger.info(f"Using ReduceLROnPlateua as a learning-rate schedule, reducing LR by 0.5 after {args.patience} steps until it reaches 1e-5.")

        elif args.lr_schedule == "cosine":
            msg = "Using a cosine annealing learning-rate schedule, "
            steps_per_cycle = args.iters_per_component if args.boosted else args.num_steps
            if args.lr_restarts > 1:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                 T_0=int(steps_per_cycle / args.lr_restarts),
                                                                                 eta_min=1e-5)
                msg += f"annealed over {steps_per_cycle}, restarting {args.lr_restarts} times within each learning cycle."

            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=steps_per_cycle,
                                                                       eta_min=1e-5)
                msg += f"annealed over {steps_per_cycle} training steps, restarting with each new component (if boosted)."

            if verbose:
                logger.info(msg)

    if args.warmup_iters > 0:
        warmup_scheduler = GradualWarmupScheduler(optimizer, total_epoch=args.warmup_iters, after_scheduler=scheduler)
        if verbose:
            logger.info(f"Gradually warming up learning rate from 0.0 to {args.learning_rate} over the first {args.warmup_iters} steps.\n")
            
        return optimizer, warmup_scheduler
    
    else:
        return optimizer, scheduler


def init_boosted_lr(model, optimizer, args):
    for c in range(args.num_components):
        # optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.
        if c != model.component:
            optimizer.param_groups[c]['lr'] = 0.0
            
    for n, param in model.named_parameters():
        param.requires_grad = True if n.startswith(f"flows.{model.component}") or n.startswith(f"flow_param.{model.component}") else False


def compute_kl_qp_loss(model, target_fn, beta, args):
    """
    Density Matching

    Compute KL(q_inv || p) where q_inv is the inverse flow transform:
    
    (log_q_inv = log_q_base - logdet),

    and p is the target distribution (energy potential)
 
    Returns the minimization objective for density matching.

    ADAPTED FROM: https://arxiv.org/pdf/1904.04676.pdf (https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py)
    """
    z0 = model.base_dist.sample((args.batch_size,))
    q_log_prob = model.base_dist.log_prob(z0).sum(1)
    
    if args.boosted:
        if model.component < model.num_components:
            density_from = '-c' if model.all_trained else '1:c-1'
            sample_from = 'c'
        else:
            density_from = '1:c'
            sample_from = '1:c'
            
        z_g, entropy_ldj, z_G, boosted_ldj = model.flow(z0, sample_from=sample_from, density_from=density_from)
        p_log_prob = -1.0 * target_fn(z_g[-1]) * beta  # p = exp(-potential) => log_p = - potential
        g_lhood = q_log_prob - entropy_ldj
        
        if model.component == 0 and model.all_trained == False:
            G_lhood = torch.zeros_like(g_lhood)
            nll = g_lhood - p_log_prob
        else:
            G_lhood = model.base_dist.log_prob(z_G[0]).sum(1) - boosted_ldj
            G_lhood = torch.max(G_lhood, torch.ones_like(G_lhood) * G_MAX_LOSS)  # when g, G not overlapping award -10 (at most) to loss
            nll = G_lhood - p_log_prob + g_lhood * args.regularization_rate

        losses = {'nll': nll.mean(), 'g_nll': g_lhood.mean().item(), 'G_nll': G_lhood.mean().item(), 'p': p_log_prob.mean().item()}

    else:
        zk, logdet = model(z0)
        p_log_prob = -1.0 * target_fn(zk) * beta  # p = exp(-potential) => log_p = - potential
        nll = q_log_prob - logdet - p_log_prob
        losses = {'nll': nll, 'q': q_log_prob.mean().item(), 'logdet': logdet.mean().item(), 'p': p_log_prob.mean().item()}

    return losses


def compute_kl_pq_loss(model, data_or_sampler, beta, args):
    """
    Density Estimation with reweighting

    Compute KL(p || q)
    """
    if callable(data_or_sampler):
        x = data_or_sampler(args.batch_size).to(args.device)
    else:
        x = data_or_sampler
    
    if args.boosted:
        if model.component > 0:

            # 1. Compute likelihood/weight for each sample
            additive = True  # recommended: True
            G_ll = torch.zeros(x.size(0))
            num_trained_components = args.num_components if model.all_trained else model.component
            for c in range(num_trained_components):
                if model.all_trained and c == model.component:
                    continue
                
                rho_simplex = model.rho[0:c+1] / torch.sum(model.rho[0:c+1])
                z_G, _, _, ldj_G, _ = model(x=x, components=c)

                if additive:
                    if c == 0:
                        G_ll = model.base_dist.log_prob(z_G).sum(1) + ldj_G
                    else:
                        last_ll = torch.log(1.0 - rho_simplex[c]) + G_ll
                        next_ll = torch.log(rho_simplex[c]) + (model.base_dist.log_prob(z_G).sum(1) + ldj_G)
                        uG_ll = torch.cat([last_ll.view(x.size(0), 1), next_ll.view(x.size(0), 1)], dim=1)
                        G_ll = torch.logsumexp(uG_ll, dim=1)
                else:
                    # multiplicative
                    G_ll += rho_simplex[c] * (model.base_dist.log_prob(z_G).sum(1) + ldj_G)
            
            G_nll = -1.0 * G_ll

            weight_samples = True  # recommended: True
            if weight_samples:
                # 2. Sample x with replacement, weighted by G_nll
                if additive:
                    weights = softmax(G_nll)
                else:
                    heuristic = "unity"
                    if heuristic == "decay":
                        beta = 1.0 / (2.0**model.component)
                    elif heuristic == "uniform":
                        beta = 1.0 / (1.0 + args.num_components)
                    else:
                        beta = 1.0  # unity
                        
                    uweights = G_nll * beta
                    weights = torch.exp(uweights - torch.logsumexp(uweights, dim=0))  # normalize weights: large NLL => large weight
                    
                weights = weights / torch.sum(weights)
                orig_weights = weights.data.clone()

                max_wt = 0.1
                if weights.max() > max_wt:
                    weights = torch.max(torch.min(weights, torch.tensor([max_wt], device=args.device)), torch.tensor([0.1 / args.batch_size], device=args.device))
                    weights = weights / torch.sum(weights)
                    
                reweighted_idx = torch.multinomial(weights, x.size(0), replacement=True)
                x_resampled = x[reweighted_idx]

                if np.random.rand() > 0.9:
                    with open(os.path.join(args.snap_dir, 'counts.txt'), 'a') as ff:
                        orig_weights.sort()
                        top_wts1 = ', '.join([f"{w:1.3f}" for w in orig_weights[-5:]])
                        weights.sort()
                        top_wts2 = ', '.join([f"{w:1.3f}" for w in weights[-5:]])
                        top_idx = ', '.join([str(ct) for _, ct in Counter(reweighted_idx.data.cpu().numpy()).most_common(10)])
                        num_unique = torch.unique(reweighted_idx).size(0)
                        print(f"C{model.component}. Unique samples={num_unique}, top ids={top_idx}, orig={top_wts1}, norm={top_wts2}", file=ff)

                # 3. Compute g for resampled observations
                z_g, _, _, ldj_g, _ = model(x=x_resampled, components="c")
                g_nll = -1.0 * (model.base_dist.log_prob(z_g).sum(1) + ldj_g)
                nll = torch.mean(g_nll)

            else:
                # Compute g in standard fashion
                z_g, _, _, ldj_g, _ = model(x=x, components="c")
                g_nll = -1.0 * (model.base_dist.log_prob(z_g).sum(1) + ldj_g)
                nll =  torch.mean(g_nll / (torch.exp(G_ll) + 0.1))

            losses = {"nll": nll}
            losses["G_nll"] = torch.mean(G_nll)
            losses["g_nll"] = torch.mean(g_nll)
            
        else:
            # train first boosted component just like a non-boosted model
            z_g, _, _, ldj_g, _ = model(x=x, components="c")
            g_nll = -1.0 * (model.base_dist.log_prob(z_g).sum(1) + ldj_g)
            losses = {"nll": torch.mean(g_nll)}
            losses["g_nll"] = torch.mean(g_nll)
            losses["G_nll"] = torch.zeros_like(losses['g_nll'])
    
    else:
        z, _, _, logdet, _ = model(x)
        q_log_prob = model.base_dist.log_prob(z).sum(1)
        nll = -1.0 * (q_log_prob + logdet)
        losses = {'nll': nll.mean(0), 'q': q_log_prob.mean().detach().item(), 'logdet': logdet.mean().detach().item()}

    return losses


@torch.no_grad()
def rho_gradient(model, target_or_sample_fn, args):
    fixed_components = "-c" if model.all_trained else "1:c-1"
    if args.density_matching:
        # density matching of a target function
        z0 = model.base_dist.sample((args.num_components * args.batch_size * 25,))
        g_zk, g_ldj = [], []
        G_zk, G_ldj = [], []
        for z0_i in z0.split(args.batch_size, dim=0):
            gZ_i, _, _, g_ldj_i = model.flow(z0_i, sample_from="c", density_from="1:c")
            g_zk += [gZ_i[-1]]  # grab K-th element
            g_ldj += [g_ldj_i]
            GZ_i, _, _, G_ldj_i = model.flow(z0_i, sample_from=fixed_components, density_from="1:c")
            G_zk += [GZ_i[-1]]  # grab K-th element
            G_ldj += [G_ldj_i]
        
        g_zk, g_ldj = torch.cat(g_zk, 0), torch.cat(g_ldj, 0)
        G_zk, G_ldj = torch.cat(G_zk, 0), torch.cat(G_ldj, 0)
        
        q_log_prob = model.base_dist.log_prob(z0).sum(1)
        p_log_prob_g = -1.0 * target_or_sample_fn(g_zk)  # p = exp(-potential) => log_p = - potential
        loss_wrt_g = q_log_prob - g_ldj - p_log_prob_g
        p_log_prob_G = -1.0 * target_or_sample_fn(G_zk)  # p = exp(-potential) => log_p = - potential
        loss_wrt_G = q_log_prob - G_ldj - p_log_prob_G
        
    else:
        # estimate density from a sampler
        sample = target_or_sample_fn(args.num_components * args.batch_size * 25).to(args.device)
        g_zk, g_ldj = [], []
        G_zk, G_ldj = [], []
        for sample_i in sample.split(args.batch_size, dim=0):
            g_zk_i, _, _, g_ldj_i = model.flow(sample_i, sample_from="c", density_from="1:c")
            g_zk += [g_zk_i[-1]]
            g_ldj += [g_ldj_i]
            G_zk_i, _, _, G_ldj_i = model.flow(sample_i, sample_from=fixed_components, density_from="1:c")
            G_zk += [G_zk_i[-1]]
            G_ldj += [G_ldj_i]

        g_zk, g_ldj = torch.cat(g_zk, 0), torch.cat(g_ldj, 0)
        G_zk, G_ldj = torch.cat(G_zk, 0), torch.cat(G_ldj, 0)  

        loss_wrt_g = -1.0 * (model.base_dist.log_prob(g_zk).sum(1) + g_ldj)
        loss_wrt_G = -1.0 * (model.base_dist.log_prob(G_zk).sum(1) + G_ldj)

    return loss_wrt_g.mean(0).detach().item(), loss_wrt_G.mean(0).detach().item()


def update_rho(model, target_or_sample_fn, writer, args):
    if model.component == 0 and model.all_trained == False:
        return

    if args.rho_iters == 0:
        return

    model.eval()
    with torch.no_grad():

        rho_log = open(model.args.snap_dir + '/rho.log', 'a')
        print(f"\n\nUpdating weight for component {model.component} (all_trained={str(model.all_trained)})", file=rho_log)
        print('Initial Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]), file=rho_log)
            
        tolerance = 0.001
        init_step_size = args.rho_lr
        min_iters = 10
        max_iters = args.rho_iters
        prev_rho = model.rho.data[model.component].item()
        
        for batch_id in range(max_iters):

            loss_wrt_g, loss_wrt_G = rho_gradient(model, target_or_sample_fn, args)            
            gradient = loss_wrt_g - loss_wrt_G

            step_size = init_step_size / (0.05 * batch_id + 1)
            rho = min(max(prev_rho - step_size * gradient, 0.0005), 0.999)

            grad_msg = f'{batch_id: >3}. rho = {prev_rho:5.3f} -  {gradient:4.2f} * {step_size:5.3f} = {rho:5.3f}'
            loss_msg = f"\tg vs G. Loss: ({loss_wrt_g:5.1f}, {loss_wrt_G:5.1f})."
            print(grad_msg + loss_msg, file=rho_log)
                    
            model.rho[model.component] = rho
            dif = abs(prev_rho - rho)
            prev_rho = rho

            writer.add_scalar(f"rho/rho_{model.component}", rho, batch_id)

            if batch_id > min_iters and (batch_id > max_iters or dif < tolerance):
                break

        print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]), file=rho_log)
        rho_log.close()


def annealing_schedule(i, args):
    if args.density_matching:
        if args.min_beta == 1.0:
            return 1.0

        if args.boosted:
            if i >= args.iters_per_component * args.num_components or i == args.iters_per_component:
                rval = 1.0
            else:
                halfway = args.iters_per_component // 2
                rval = 0.01 + ((i % halfway) / halfway) if (i % args.iters_per_component) < halfway else 1.0
        else:
            rval = 0.01 + i/10000.0

        rval = max(args.min_beta, min(args.max_beta, rval))
    else:
        rval = 1.0
            
    return rval


def train(model, target_or_sample_fn, loss_fn, optimizer, scheduler, args):
    if args.tensorboard:
        writer = SummaryWriter(args.snap_dir)

    model.train()

    if args.boosted:
        model.component = 0
        init_boosted_lr(model, optimizer, args)
    
    for batch_id in range(args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        beta = annealing_schedule(batch_id, args)

        if args.dataset == "u0" and batch_id == args.iters_per_component + 1:
            args.dataset = "u1"
            target_or_sample_fn = make_toy_density(args)

        losses = loss_fn(model, target_or_sample_fn, beta, args)
        losses['nll'].backward()

        if args.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_clip)
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if args.tensorboard:
                writer.add_scalar("grad_norm/grad_norm", grad_norm, batch_id)

        if args.boosted:  # freeze all but the new component being trained
            if batch_id > 0:
                for c in range(args.num_components):
                    if c != model.component:
                        optimizer.param_groups[c]['lr'] = 0.0
        if args.tensorboard:
            for i in range(len(optimizer.param_groups)):
                writer.add_scalar(f'lr/lr_{i}', optimizer.param_groups[i]['lr'], batch_id)

        optimizer.step()
        if not args.no_lr_schedule:
            if args.lr_schedule == "plateau":
                scheduler.step(metrics=losses['nll'])
            else:
                scheduler.step()

        boosted_component_converged = args.boosted and batch_id % args.iters_per_component == 0 and batch_id > 0
        new_boosted_component = args.boosted and batch_id % args.iters_per_component == 1
        if boosted_component_converged or new_boosted_component or batch_id % args.log_interval == 0:
            msg = f"{args.dataset}: step {batch_id:5d} / {args.num_steps}; loss {losses['nll'].item():8.3f} (beta={beta:4.2f})"
            if args.boosted:
                msg += f" | g vs G ({losses['g_nll']:8.3f}, {losses['G_nll']:8.3f})"
                msg += f" | p_log_prob {losses['p']:8.3f}" if args.density_matching else ''
                msg += f" | c={model.component} (all={str(model.all_trained)[0]})"
                msg += f" | Rho=[" + ', '.join([f"{val:4.2f}" for val in model.rho.data]) + "]"
            else:
                msg += f" | q_log_prob {losses['q']:8.3f}"
                msg += f" | ldj {losses['logdet']:8.3f}"
                msg += f" | p_log_prob {losses['p']:7.3f}" if args.density_matching else ''
            logger.info(msg)

        if args.tensorboard:
            writer.add_scalar('batch/train_loss', losses['nll'].item(), batch_id)
            if args.boosted:
                writer.add_scalar('batch/train_g', losses['g_nll'], batch_id)
                writer.add_scalar('batch/train_G', losses['G_nll'], batch_id)
            else:
                writer.add_scalar('batch/q_log_prob', losses['q'], batch_id)
                writer.add_scalar('batch/log_det_jacobian', losses['logdet'], batch_id)

        if boosted_component_converged:
            update_rho(model, target_or_sample_fn, writer, args)
            model.increment_component()
            optimizer, scheduler = init_optimizer(model, args, verbose=False)
            init_boosted_lr(model, optimizer, args)

        if (batch_id > 0 and batch_id % args.plot_interval == 0) or boosted_component_converged:
            with torch.no_grad():
                plot(batch_id, model, target_or_sample_fn, args)

 
def main(main_args=None):
    """
    use main_args to run this script as function in another script
    """

    # =========================================================================
    # PARSE EXPERIMENT SETTINGS, SETUP SNAPSHOTS DIRECTORY, LOGGING
    # =========================================================================
    args, kwargs = parse_args(main_args)

    # =========================================================================
    # SAVE EXPERIMENT SETTINGS
    # =========================================================================
    logger.info(f'EXPERIMENT SETTINGS:\n{args}\n')
    torch.save(args, os.path.join(args.snap_dir, 'config.pt'))

    # =========================================================================
    # INITIALIZE MODEL AND OPTIMIZATION
    # =========================================================================
    model = init_model(args)
    optimizer, scheduler = init_optimizer(model, args)
    num_params = sum([param.nelement() for param in model.parameters()])    
    logger.info(f"MODEL:\nNumber of model parameters={num_params}\n{model}\n")

    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info('TRAINING:')
    if args.density_matching:
        # target is energy potential to match
        target_or_sample_fn = make_toy_density(args)
        loss_fn = compute_kl_qp_loss
    else:
        # target is density to estimate to sample from
        target_or_sample_fn = make_toy_sampler(args)
        loss_fn = compute_kl_pq_loss


    train(model, target_or_sample_fn, loss_fn, optimizer, scheduler, args)
        

if __name__ == "__main__":
    main()
