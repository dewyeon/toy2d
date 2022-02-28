import torch
from functools import reduce
import numpy as np
import random
import datetime
import time
from scipy.special import logsumexp
import logging

from optimization.loss import calculate_loss, calculate_loss_array
from utils.plotting import plot_reconstructions, plot_decoded_random_sample, plot_decoded_manifold, plot_data_manifold, plot_flow_samples

logger = logging.getLogger(__name__)


def evaluate(data_loader, model, args, epoch=None, results_type=None):
    """
    data_loader:  pytorch data loader
    model:        a pytorch model
    args:         command line arguments
    epoch:        Current epoch in training, used to control how often plots of reconstructions are saved
    results_type: String describing the type of results (e.g. 'Valdiation' or 'Test'). The final
                  validation loss computed will only be printed if results_type is not None, similarly
                  plots and evaluation information is only created if results_type is not None.
    """
    model.eval()
    save_this_epoch = epoch is None or epoch==1 or (args.plot_interval > 0 and epoch % args.plot_interval == 0)
    loss = 0.0
    rec = 0.0
    kl = 0.0

    for batch_id, (x, _) in enumerate(data_loader):
        x = x.to(args.device)

        if args.vae_layers == 'linear':
            x = x.view(-1, np.prod(args.input_size))
        else:
            x = x.view(-1, *args.input_size)

        if args.flow == 'boosted':
            if model.component == 0 and not model.all_trained:
                # only one component trained so no need for multiple sampling
                x_recon, z_mu, z_var, Z, ldj, _, _ = model(x, prob_all=1.0)
                z0, zk = Z[0], Z[-1]
            else:
                # take multiple samples from the mixture model
                num_repeats = args.num_components * 3

                h, z_mu, z_var = model.encode(x)
                z_0 = model.reparameterize(z_mu, z_var)

                x_recon, zk, ldj = [], [], []
                for i in range(num_repeats):
                    zk_i, ldj_i, _, _ = model.flow(z_0, sample_from="1:c", density_from=None, h=h)
                    zk += [zk_i[-1]]
                    ldj += [ldj_i]
                    x_recon_i = model.decode(zk_i[-1])
                    x_recon += [x_recon_i]

                x = torch.cat(num_repeats * [x])
                z_mu = torch.cat(num_repeats * [z_mu])
                z_var = torch.cat(num_repeats * [z_var])
                z0 = torch.cat(num_repeats * [z_0])
                zk = torch.cat(zk, 0)
                ldj = torch.cat(ldj, 0)
                x_recon = torch.cat(x_recon, 0)
        else:
            x_recon, z_mu, z_var, ldj, z0, zk = model(x)
            
        batch_loss, batch_rec, batch_kl = calculate_loss(x_recon, x, z_mu, z_var, z0, zk, ldj, args)
        loss += batch_loss.item()
        rec += batch_rec.item()
        kl += batch_kl.item()

        # Plots reconstructions
        if batch_id == 0 and save_this_epoch and args.save_results:
            plot_reconstructions(data=x, recon_mean=x_recon, loss=batch_loss, args=args, epoch=epoch)
            
    if model.z_size == 2 and save_this_epoch and args.save_results:
        plot_flow_samples(epoch, model, data_loader, args)

    avg_loss = loss / len(data_loader)
    avg_rec = rec / len(data_loader)
    avg_kl = kl / len(data_loader)

    if results_type is not None:

        if args.dataset == "mnist":
            plot_decoded_random_sample(args, model, size_x=5, size_y=5)
            if model.z_size == 2:
                plot_decoded_manifold(model, args)
                plot_data_manifold(model, data_loader, args)

        results_msg = f'{results_type} set loss: {avg_loss:.4f}, Reconstruction: {avg_rec:.4f}, KL-Divergence: {avg_kl:.4f}'
        results_msg += f', BPD: {avg_loss / (np.prod(args.input_size) * np.log(2.0)):.4f}' if args.input_type != "binary" else ''
        logger.info(results_msg + "\n")

        if args.save_results:
            with open(args.exp_log, 'a') as ff:
                print(results_msg, file=ff)

    return avg_loss, avg_rec, avg_kl


def evaluate_likelihood(data_loader, model, args, S=5000, MB=1000, results_type=None):
    """
    Calculate negative log likelihood using importance sampling
    """
    model.eval()

    X = torch.cat([x for x, y in list(data_loader)], 0).to(args.device)

    # set auxiliary variables for number of training and test sets
    N_test = X.size(0)

    likelihood_test = []

    if S <= MB:
        R = 1
    else:
        R = S // MB
        S = MB

    for j in range(N_test):
        #if j % (int(N_test / 10.0)) == 0:
        #    logger.info('Progress: {:.1f}%'.format(100.0 * j / (1.0 * N_test))) 

        x_single = X[j].unsqueeze(0)

        if args.vae_layers == 'linear':
            x_single = x_single.view(-1, np.prod(args.input_size))
        else:
            x_single = x_single.view(-1, *args.input_size)

        a = []
        for r in range(0, R):
            # Repeat it for all training points
            x = x_single.expand(S, *x_single.size()[1:]).contiguous()

            if args.flow == 'boosted':
                x_mean, z_mu, z_var, Z, ldj, _, _ = model(x, prob_all=1.0)
                z0, zk = Z[0], Z[-1]
            else:
                x_mean, z_mu, z_var, ldj, z0, zk = model(x)
            
            a_tmp = calculate_loss_array(x_mean, x, z_mu, z_var, z0, zk, ldj, args)
            a.append(-a_tmp.cpu().data.numpy())

        # calculate max
        a = np.asarray(a)
        a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
        likelihood_x = logsumexp(a)
        likelihood_test.append(likelihood_x - np.log(len(a)))

    likelihood_test = np.array(likelihood_test)
    nll = -np.mean(likelihood_test)

    if args.save_results:
        results_msg = f'{results_type} set NLL: {nll:.4f}'
        if args.input_type != 'binary':
            bpd = nll / (np.prod(args.input_size) * np.log(2.))
            results_msg += f', NLL BPD: {bpd:.4f}'
        results_msg += '\n'

        logger.info(results_msg)

        with open(args.exp_log, 'a') as ff:
            print(results_msg, file=ff)

    return nll

