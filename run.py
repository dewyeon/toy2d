import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n

#['moons', 'pinwheel', 'checkerboard', '2spirals']
TOY_DATASETS = ["8gaussians", "2gaussians", "swissroll", "checkerboard", "joint_gaussian"]

if args.n == 0:
    for num_flows in [8,16,24]:
        for toy_exp_type in [0]:
            for dataset in ["pinwheel", "cos", "2spirals", "checkerboard", "line", "circles", "joint_gaussian"]:
                subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)} --gpu_id {n}", shell=True)

if args.n == 1:
    for num_flows in [8,16,24]:
        for toy_exp_type in [0]:
            for dataset in ["8gaussians", "2gaussians", "1gaussian",  "swissroll", "rings", "moons"]:
                subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)} --gpu_id {n}", shell=True)
        
if args.n == 2:
    for num_flows in [8,16,24]:
        for toy_exp_type in [1]:
            for sampling_num in [256, 4096]:
                for dataset in TOY_DATASETS:
                    for norm_hyp in [1, 0.1]:    
                        subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --sampling_num {sampling_num} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)+'_'+str(norm_hyp)+'_'+str(sampling_num)} --gpu_id {n}", shell=True)

if args.n == 3:
    for num_flows in [8,16,24]:
        for toy_exp_type in [1]:
            for sampling_num in [256, 4096]:
                for dataset in TOY_DATASETS:
                    for norm_hyp in [0.01, 0.001]:    
                        subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --sampling_num {sampling_num} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)+'_'+str(norm_hyp)+'_'+str(sampling_num)} --gpu_id {n}", shell=True)

if args.n == 4:
    for num_flows in [8,16,24]:
        for toy_exp_type in [1]:
            for sampling_num in [256, 4096]:
                for dataset in TOY_DATASETS:
                    for norm_hyp in [0.0001, 0.00001]:    
                        subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --sampling_num {sampling_num} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)+'_'+str(norm_hyp)+'_'+str(sampling_num)} --gpu_id {n}", shell=True)

if args.n == 5:
    for num_flows in [8,16,24]:
        for toy_exp_type in [2]:
            for sampling_num in [256, 4096]:
                for dataset in TOY_DATASETS:
                    for norm_hyp in [1, 0.1]:    
                        subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --sampling_num {sampling_num} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)+'_'+str(norm_hyp)+'_'+str(sampling_num)} --gpu_id {n}", shell=True)

if args.n == 6:
    for num_flows in [8,16,24]:
        for toy_exp_type in [2]:
            for sampling_num in [256, 4096]:
                for dataset in TOY_DATASETS:
                    for norm_hyp in [0.01, 0.001]:    
                        subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --sampling_num {sampling_num} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)+'_'+str(norm_hyp)+'_'+str(sampling_num)} --gpu_id {n}", shell=True)

if args.n == 7:
    for num_flows in [8,16,24]:
        for toy_exp_type in [2]:
            for sampling_num in [256, 4096]:
                for dataset in TOY_DATASETS:
                    for norm_hyp in [0.0001, 0.00001]:    
                        subprocess.call(f"python realnvp_toy_experiment.py --use_wandb 'True' --norm_hyp {norm_hyp} --sampling_num {sampling_num} --toy_exp_type {toy_exp_type} --no_tensorboard --dataset {dataset} --num_flows {num_flows} --learning_rate 1e-4 --experiment_name {'[Exp'+str(toy_exp_type)+']K'+str(num_flows)+'_realnvp'+str(dataset)+'_'+str(norm_hyp)+'_'+str(sampling_num)} --gpu_id {n}", shell=True)

