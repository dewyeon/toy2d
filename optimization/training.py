import torch
from functools import reduce
import numpy as np
import random
import datetime
import time
import logging
from shutil import copyfile
from tensorboardX import SummaryWriter

from optimization.loss import calculate_loss, calculate_loss_array, calculate_boosted_loss
from utils.plotting import plot_training_curve
from optimization.evaluation import evaluate
from optimization.optimizers import init_optimizer
from utils.utilities import save, load

logger = logging.getLogger(__name__)


def train(train_loader, val_loader, model, optimizer, scheduler, args):
    if args.tensorboard:
        args.writer = SummaryWriter(args.snap_dir)

    header_msg = f'| Epoch |  TRAIN{"Loss": >12}{"Reconstruction": >18}'
    header_msg += f'{"Log G": >12}{"Prior": >12}{"Entropy": >12}{"Log Ratio": >12}{"| ": >4}' if args.flow == "boosted" else f'{"KL": >12}{"| ": >4}'
    header_msg += f'{"VALIDATION": >11}{"Loss": >12}{"Reconstruction": >18}{"KL": >12}{"| ": >4}{"Annealing": >12}'
    header_msg += f'{"P(c in 1:C)": >16}{"Component": >10}{"Improved": >10}{"|": >3}' if args.flow == "boosted" else f'{"|": >3}'
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    logger.info(header_msg)
    logger.info('|' + "-"*(len(header_msg)-2) + '|')

    if args.flow == "boosted":
        t_loss, t_rec, t_G, t_p, t_entropy, v_loss, v_rec, v_kl, t_times  = train_boosted(
            train_loader, val_loader, model, optimizer, scheduler, args)
    else:
        t_loss, t_rec, t_kl, v_loss, v_rec, v_kl, t_times  = train_vae(
            train_loader, val_loader, model, optimizer, scheduler, args)

    # save training and validation results
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    timing_msg = f"\nStopped after {t_times.shape[0]} epochs"
    timing_msg += f"\nAverage train time per epoch: {np.mean(t_times):.2f} +/- {np.std(t_times, ddof=1):.2f}\n"
    logger.info(timing_msg)

    if args.save_results:
        if args.flow == "boosted":
            np.savetxt(args.snap_dir + '/train_loss.csv', t_loss, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_rec.csv', t_rec, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_log_G_z.csv', t_G, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_log_p_zk.csv', t_p, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_entropy.csv', t_entropy, fmt='%f', delimiter=',')
        else:
            np.savetxt(args.snap_dir + '/train_loss.csv', t_loss, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_rec.csv', t_rec, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_kl.csv', t_kl, fmt='%f', delimiter=',')
        
        np.savetxt(args.snap_dir + '/val_loss.csv', v_loss, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/val_rec.csv', v_rec, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/val_kl.csv', v_kl, fmt='%f', delimiter=',')

        plot_training_curve(t_loss, v_loss, fname=args.snap_dir + 'training_curve.png')

        with open(args.exp_log, 'a') as ff:
            timestamp = str(datetime.datetime.now())[0:19].replace(' ', '_')
            setup_msg = ' '.join([timestamp, args.flow, args.dataset]) + "\n" + repr(args)
            print("\n" + setup_msg + "\n" + timing_msg, file=ff)

    return t_loss, v_loss


def train_vae(train_loader, val_loader, model, optimizer, scheduler, args):
    train_loss = []
    train_rec = []
    train_kl = []
    val_loss = []
    val_rec = []
    val_kl = []

    # for early stopping
    best_loss = np.inf
    best_bpd = np.inf
    e = 0
    epoch = 0
    train_times = []

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        tr_loss, tr_rec, tr_kl = train_epoch_vae(epoch, train_loader, model, optimizer, scheduler, args)    
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        beta = max(min([(epoch * 1.) / max([args.annealing_schedule, 1.]), args.max_beta]), args.min_beta)
        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}{"| ": >4}'
        logger.info(epoch_msg)
        if args.tensorboard:
            args.writer.add_scalar('epoch/train_loss', tr_loss.mean(), epoch)
            args.writer.add_scalar('epoch/train_rec', tr_rec.mean(), epoch)
            args.writer.add_scalar('epoch/train_kl', tr_kl.mean(), epoch)
            args.writer.add_scalar('epoch/valid_loss', v_loss, epoch)
            args.writer.add_scalar('epoch/valid_rec', v_rec, epoch)
            args.writer.add_scalar('epoch/valid_kl', v_kl, epoch)
            args.writer.add_scalar('epoch/beta', beta, epoch)

        # early-stopping: does adding a new component help?
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            save(model, optimizer, args.snap_dir + 'model.pt', scheduler)
        elif (args.early_stopping_epochs > 0) and (epoch >= args.annealing_schedule):
            e += 1
            if e > args.early_stopping_epochs:
                break

    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_kl = np.hstack(train_kl)
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_kl, val_loss, val_rec, val_kl, train_times


def train_epoch_vae(epoch, train_loader, model, optimizer, scheduler, args):
    model.train()
    num_trained = 0
    total_samples = len(train_loader.sampler)
    total_batches = len(train_loader)
    train_loss = np.zeros(total_batches)
    train_bpd = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_kl = np.zeros(total_batches)

    # set beta annealing coefficient
    beta = max(min([(epoch * 1.) / max([args.annealing_schedule, 1.]), args.max_beta]), args.min_beta) if args.load is not None else 1.0
    step = (epoch - 1) * total_batches
    for batch_id, (x, _) in enumerate(train_loader):
        x = x.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(x)

        if args.vae_layers == 'linear':
            x = x.view(-1, np.prod(args.input_size))
        else:
            x = x.view(-1, *args.input_size)

        optimizer.zero_grad()
        x_mean, z_mu, z_var, ldj, z0, zk = model(x)
        loss, rec, kl = calculate_loss(x_mean, x, z_mu, z_var, z0, zk, ldj, args, beta=beta)
        loss.backward()

        if args.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_clip)
        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            args.writer.add_scalar("grad_norm/grad_norm", grad_norm, step)

        if args.tensorboard:
            for i in range(len(optimizer.param_groups)):
                args.writer.add_scalar(f'lr/lr_{i}', optimizer.param_groups[i]['lr'], step)
        optimizer.step()
        if not args.no_lr_schedule:
            if args.lr_schedule == "plateau":
                scheduler.step(metrics=loss)
            else:
                scheduler.step()

        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_kl[batch_id] = kl.item()
        step += 1
        if args.tensorboard:
            args.writer.add_scalar('step_loss/train_loss', loss.item(), step)
            args.writer.add_scalar('step_loss/train_rec', rec.item(), step)
            args.writer.add_scalar('step_loss/train_kl', kl.item(), step)

    return train_loss, train_rec, train_kl


def train_boosted(train_loader, val_loader, model, optimizer, scheduler, args):    
    train_times = []
    train_loss = []
    train_rec = []
    train_G = []
    train_p = []
    train_entropy = []
    
    val_loss = []
    val_rec = []
    val_kl = []

    # for early stopping
    best_loss = np.array([np.inf] * args.num_components)
    best_tr_ratio = np.array([-np.inf] * args.num_components)
    early_stop_count = 0
    converged_epoch = 0  # corrects the annealing schedule when a component converges early
    v_loss = 9999999.9

    # initialize learning rates for boosted components
    prev_lr = init_boosted_lr(model, optimizer, args)

    args.step = 0
    for epoch in range(args.init_epoch, args.epochs + 1):

        # compute annealing rate for KL loss term
        beta = kl_annealing_rate(epoch - converged_epoch, model.component, model.all_trained, args)

        # occasionally sample from all components to keep decoder from focusing solely on new component
        prob_all = sample_from_all_prob(epoch - converged_epoch, model.component, model.all_trained, args)

        # Train model
        t_start = time.time()
        tr_loss, tr_rec, tr_G, tr_p, tr_entropy, tr_ratio, prev_lr = train_epoch_boosted(
            epoch, train_loader, model, optimizer, scheduler, beta, prob_all, prev_lr, v_loss, args)
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_G.append(tr_G)
        train_p.append(tr_p)
        train_entropy.append(tr_entropy)

        # Evaluate model
        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)
        
        # Assess convergence
        component_converged, model_improved, early_stop_count, best_loss, best_tr_ratio = check_convergence(
            early_stop_count, v_loss, best_loss, tr_ratio, best_tr_ratio, epoch - converged_epoch, model, args)

        # epoch level reporting
        epoch_msg = epoch_reporting(model, tr_loss, tr_rec, tr_G, tr_p, tr_entropy, tr_ratio, v_loss, v_rec, v_kl, beta, prob_all, train_times, epoch, model_improved, args)

        if model_improved:
            fname = f'model_c{model.component}.pt' if args.boosted and args.save_intermediate_checkpoints else 'model.pt'
            save(model, optimizer, args.snap_dir + fname, scheduler)
            
        if component_converged:
            logger.info(epoch_msg + f'{"| ": >4}')
            logger.info("-" * 206)
            converged_epoch = epoch

            # revert back to the last best version of the model and update rho
            fname = f'model_c{model.component}.pt' if args.save_intermediate_checkpoints else 'model.pt'
            load(model=model, optimizer=optimizer, path=args.snap_dir + fname, args=args, scheduler=scheduler, verbose=False)
            model.update_rho(train_loader)
            
            last_component = model.component == (args.num_components - 1)
            no_fine_tuning = args.epochs <= args.epochs_per_component * args.num_components
            fine_tuning_done = model.all_trained and last_component
            if (fine_tuning_done or no_fine_tuning) and last_component:
                # stop the full model after all components have been trained
                logger.info(f"Model converged, training complete, saving: {args.snap_dir + 'model.pt'}")
                model.all_trained = True
                save(model, optimizer, args.snap_dir + f'model.pt', scheduler)
                break

            save(model, optimizer, args.snap_dir + f'model_c{model.component}.pt', scheduler)
            
            # reset early_stop_count and train the next component
            model.increment_component()
            early_stop_count = 0
            v_loss = 9999999.9
            optimizer, scheduler = init_optimizer(model, args, verbose=False)
            prev_lr = init_boosted_lr(model, optimizer, args)
        else:
            logger.info(epoch_msg + f'{"| ": >4}')
            if epoch == args.epochs:
                if args.boosted and args.save_intermediate_checkpoints:
                    # Save the best version of the model trained up to the current component with filename model.pt
                    # This is to protect against times when the model is trained/re-trained but doesn't run long enough
                    #   for all components to converge / train completely
                    copyfile(args.snap_dir + f'model_c{model.component}.pt', args.snap_dir + 'model.pt')
                    logger.info(f"Resaving last improved version of {f'model_c{model.component}.pt'} as 'model.pt' for future testing")
                else:
                    logger.info(f"Stopping training after {epoch} epochs of training.")
        
    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_G = np.hstack(train_G)
    train_p = np.hstack(train_p)
    train_entropy = np.hstack(train_entropy)
    
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_G, train_p, train_entropy, val_loss, val_rec, val_kl, train_times


def train_epoch_boosted(epoch, train_loader, model, optimizer, scheduler, beta, prob_all, prev_lr, v_loss, args):
    model.train()
    is_first_component = model.component == 0 and not model.all_trained
    
    total_batches = len(train_loader)
    total_samples = len(train_loader.sampler) * 1.0
    train_loss = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_p = np.zeros(total_batches)
    train_entropy = np.zeros(total_batches)
    train_G = []
    train_ratio = []
    grad_norm = None

    for batch_id, (x, _) in enumerate(train_loader):
        #step = (epoch - 1) * total_batches + batch_id
        x = x.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(x)

        if args.vae_layers == 'linear':
            x = x.view(-1, np.prod(args.input_size))
        else:
            x = x.view(-1, *args.input_size)

        optimizer.zero_grad()
        x_recon, z_mu, z_var, z_g, g_ldj, z_G, G_ldj = model(x, prob_all=prob_all)

        loss, rec, log_G, log_p, entropy, log_ratio = calculate_boosted_loss(
            x_recon, x, z_mu, z_var, z_g, g_ldj, z_G, G_ldj, args, is_first_component, beta)
        loss.backward()

        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Adjust learning rates for boosted model, keep fixed components frozen
        update_learning_rates(prev_lr, model.component, optimizer, args)

        losses = {'loss': loss, 'rec': rec, 'log_p': log_p, 'entropy': entropy}
        batch_reporting(optimizer, losses, grad_norm, args)

        # Perform gradient update, modify learning rate according to learning rate schedule
        optimizer.step()
        if not args.no_lr_schedule:
            prev_lr = update_scheduler(prev_lr, model.component, optimizer, scheduler, v_loss, args)

        # save results
        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_p[batch_id] = log_p.item()
        train_entropy[batch_id] = entropy.item()
        args.step += 1

        # ignore the boosting terms if we sampled from all (to alleviate decoder shock)
        if z_G is not None and G_ldj is not None:
            train_G.append(log_G.item())
            train_ratio.append(log_ratio.item())

    train_G = np.array(train_G) if len(train_G) > 0 else np.zeros(1)
    train_ratio = np.array(train_ratio) if len(train_ratio) > 0 else np.zeros(1)
    return train_loss, train_rec, train_G, train_p, train_entropy, (train_ratio.sum() / total_samples), prev_lr


def epoch_reporting(model, tr_loss, tr_rec, tr_G, tr_p, tr_entropy, tr_ratio, v_loss, v_rec, v_kl, beta, prob_all, train_times, epoch, model_improved, args):
    epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_G.mean():12.3f}{tr_p.mean():12.3f}{tr_entropy.mean():12.3f}{tr_ratio:12.3f}'
    epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}{prob_all:16.2f}{model.component:10d}'
    epoch_msg += f'{"T": >10}' if model_improved else ' '*10
    
    if args.tensorboard:
        args.writer.add_scalar('epoch/train_loss', tr_loss.mean(), epoch)
        args.writer.add_scalar('epoch/train_rec', tr_rec.mean(), epoch)
        args.writer.add_scalar('epoch/train_G', tr_G.mean(), epoch)
        args.writer.add_scalar('epoch/train_p', tr_p.mean(), epoch)
        args.writer.add_scalar('epoch/train_entropy', tr_entropy.mean(), epoch)
        args.writer.add_scalar('epoch/train_G_vs_g_ratio', tr_ratio, epoch)
        args.writer.add_scalar('epoch/valid_loss', v_loss, epoch)
        args.writer.add_scalar('epoch/valid_rec', v_rec, epoch)
        args.writer.add_scalar('epoch/valid_kl', v_kl, epoch)
        args.writer.add_scalar('epoch/beta', beta, epoch)
        args.writer.add_scalar('epoch/prob_all', prob_all, epoch)
        args.writer.add_scalar('epoch/train_times', train_times[-1], epoch)

    return epoch_msg


def batch_reporting(optimizer, losses, grad_norm, args):
    if args.tensorboard:
        args.writer.add_scalar('step_loss/train_loss', losses['loss'].item(), args.step)
        args.writer.add_scalar('step_loss/train_rec', losses['rec'].item(), args.step)
        args.writer.add_scalar('step_loss/train_logp', losses['log_p'].item(), args.step)
        args.writer.add_scalar('step_loss/train_entropy', losses['entropy'].item(), args.step)
        
        for i in range(len(optimizer.param_groups)):
            args.writer.add_scalar(f'lr/lr_{i}', optimizer.param_groups[i]['lr'], args.step)
        
        if args.max_grad_norm > 0:
            args.writer.add_scalar("grad_norm/grad_norm", grad_norm, args.step)


def update_learning_rates(prev_lr, component, optimizer, args):
    for c in range(args.num_components):
        optimizer.param_groups[c]['lr'] = prev_lr[c] if c == component else 0.0


def update_scheduler(prev_lr, component, optimizer, scheduler, loss, args):
    if args.lr_schedule == "plateau":
        scheduler.step(metrics=loss)
    else:
        scheduler.step()
        
    if args.boosted:
        prev_lr[component] = optimizer.param_groups[component]['lr']
    else:
        prev_lr = []

    return prev_lr


def init_boosted_lr(model, optimizer, args):
    learning_rates = []
    for c in range(args.num_components):
        if c != model.component:
            optimizer.param_groups[c]['lr'] = 0.0
            
        learning_rates.append(optimizer.param_groups[c]['lr'])

    for n, param in model.named_parameters():
        param.requires_grad = True if n.startswith(f"flow_param.{model.component}") or not n.startswith("flow_param") else False

    return learning_rates


def kl_annealing_rate(epochs_since_prev_convergence, component, all_trained, args):
    """
    TODO need to adjust this for when an previous component converged early
    """
    past_warmup =  ((epochs_since_prev_convergence - 1) % args.epochs_per_component) >= args.annealing_schedule
    if all_trained or past_warmup:
        # all trained or past the first args.annealing_schedule epochs of training this component, so no annealing
        beta = args.max_beta
    else:
        # within the first args.annealing_schedule epochs of training this component
        beta = (((epochs_since_prev_convergence - 1) % args.annealing_schedule) / args.annealing_schedule) * args.max_beta
        beta += 1.0 / args.annealing_schedule  # don't want annealing rate to start at zero
            
    beta = min(beta, args.max_beta)
    beta = max(beta, args.min_beta)
    return beta


def sample_from_all_prob(epochs_since_prev_convergence, current_component, all_trained, args):
    """
    Want to occasionally sample from all components so decoder doesn't solely focus on new component
    """
    max_prob_all = min(0.5, 1.0 - (1.0 / (args.num_components)))
    min_prob_all = 0.1
    if all_trained:
        # all components trained and rho updated for all components, make sure annealing rate doesn't continue to cycle
        return max_prob_all

    else:
        if current_component == 0:
            return 0.0
        else:
            pct_trained = ((epochs_since_prev_convergence - 1) % args.epochs_per_component) / args.epochs_per_component
            pct_trained += (1.0 / args.epochs_per_component)  # non-zero offset (don't start at zero)
            prob_all = max(min_prob_all, min(pct_trained, 1.0) * max_prob_all)
            
        return prob_all


def check_convergence(early_stop_count, v_loss, best_loss, tr_ratio, best_tr_ratio, epochs_since_prev_convergence, model, args):
    """
    Verify if a boosted component has converged
    """
    c = model.component
    first_component_trained = model.component > 0 or model.all_trained
    model_improved = v_loss < best_loss[c]
    early_stop_flag = False
    if first_component_trained and v_loss < best_loss[c]: # tried also checking: tr_ratio > best_tr_ratio[c]), but simpler is better
        # already trained more than one component, boosted component improved
        early_stop_count = 0
        best_loss[c] = v_loss
        best_tr_ratio[c] = tr_ratio
    elif not first_component_trained and v_loss < best_loss[c]:
        # training only the first component (for the first time), and it improved
        early_stop_count = 0
        best_loss[c] = v_loss
    elif args.early_stopping_epochs > 0:
        # model didn't improve, do we consider it converged yet?
        early_stop_count += 1        
        early_stop_flag = early_stop_count > args.early_stopping_epochs

    # Lastly, we consider the model converged if a pre-set number of epochs have elapsed
    time_to_update = epochs_since_prev_convergence % args.epochs_per_component == 0

    # But, model must have exceeded the warmup period before "converging"
    past_warmup = (epochs_since_prev_convergence >= args.annealing_schedule) or model.all_trained
    
    converged = (early_stop_flag or time_to_update) and past_warmup
    return converged, model_improved, early_stop_count, best_loss, best_tr_ratio

