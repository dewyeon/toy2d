import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n

#['moons', 'pinwheel', 'checkerboard', '2spirals']
TOY_DATASETS = ["8gaussians", "2gaussians", "swissroll", "checkerboard", "joint_gaussian"]
# python realnvp_toy_experiment.py --toy_exp_type 1 --no_tensorboard --dataset 8gaussians --num_flows 2 --learning_rate 1e-4 --norm_hyp 0 --kde_bandwidth 0.1


NUM_FLOWS = 2
KDE_BANDWIDTH = 0.6
DATASET = "checkerboard"

# toy_exp_type = ['default', 'KL(f|g)', 'KL(g|f)']

# if args.n == 6:
#     for toy_exp_type in ['default']:
#         for norm_hyp in [0]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 1:
#     for toy_exp_type in ['KL_fg']:
#         for norm_hyp in [0.1]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 1:
#     for toy_exp_type in ['KL_gf']:
#         for norm_hyp in [1000]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 2:
#     for toy_exp_type in ['norm_density']:
#         for norm_hyp in [0.01]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 1:
#     for toy_exp_type in ['norm_log_density']:
#         for norm_hyp in [0.0001]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 1:
#     for toy_exp_type in ['default']:
#         for norm_hyp in [0]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 2:
#     for toy_exp_type in ['KL_-fg']:
#         for norm_hyp in [0.1]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

if args.n == 0:
    for toy_exp_type in ['default']:
        for norm_hyp in [0.1]:
            subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

if args.n == 1:
    for toy_exp_type in ['KL_gf']:
        for norm_hyp in [0.1]:
            subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

if args.n == 2:
    for toy_exp_type in ['KL_g-f']:
        for norm_hyp in [0.1]:
            subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 4:
#     for toy_exp_type in ['absKL_fg']:
#         for norm_hyp in [0.1]:
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

if args.n == 3:
    for toy_exp_type in ['absKL_gf']:
        for norm_hyp in [0.1]:
            subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

# if args.n == 6:
#     for toy_exp_type in ['absKL_-fg']:
#         for norm_hyp in [0.1]: 
#             subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)

if args.n == 4:
    for toy_exp_type in ['absKL_g-f']:
        for norm_hyp in [0.1]:
            subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --kde_bandwidth {KDE_BANDWIDTH} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {DATASET} --num_flows {NUM_FLOWS} --learning_rate 1e-4 --experiment_name {'[Exp'+toy_exp_type+']K'+str(NUM_FLOWS)+'_realnvp_KDE'+str(KDE_BANDWIDTH) +DATASET+'_'+str(norm_hyp)} --gpu_id {n}", shell=True)


# default: K=2, 8
# KL_fg: K=8, 1,10,100
# KL_gf: K=8, 1,10,100,1000
# KL_g_minusf: K=8, 1,
# KL_g_newf: K=8, 1,
# KL_g_newminusf: K=8, 1,