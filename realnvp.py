import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n

# if args.n == 1:
#     for dataset in ['checkerboard', '2spirals', 'moons', 'pinwheel', '8gaussians']:
#         subprocess.call(f"python realnvp_toy_experiment.py --no_tensorboard --dataset {dataset} --num_flows 4 --learning_rate 1e-4 --experiment_name {'k4_realnvp'+str(dataset)} --gpu_id {n}", shell=True)

# if args.n == 2:
#     for dataset in ['checkerboard', '2spirals', 'moons', 'pinwheel', '8gaussians']:
#         subprocess.call(f"python realnvp_toy_experiment.py --no_tensorboard --dataset {dataset} --num_flows 8 --learning_rate 1e-4 --experiment_name {'k8_realnvp'+str(dataset)} --gpu_id {n}", shell=True)




conda activate nf; cd nf/toy2d
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name '[base]k8_realnvp_8gaussians' --norm_hyp 0 --gpu_id 0
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.01 --gpu_id 0
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.005 --gpu_id 0
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.002 --gpu_id 1
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.001 --gpu_id 1
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.0005 --gpu_id 2
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.0002 --gpu_id 2
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.0001 --gpu_id 3
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --norm_hyp 0.00005 --gpu_id 3
# python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '2gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_2gaussians' --norm_hyp 0.00002 --gpu_id 4
# python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '2gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_2gaussians' --norm_hyp 0.00001 --gpu_id 4

python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name '[base]k8_realnvp_checkerboard' --norm_hyp 0 --gpu_id 4
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.01 --gpu_id 4
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.005 --gpu_id 4
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.002 --gpu_id 5
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.001 --gpu_id 5
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.0005 --gpu_id 6
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.0002 --gpu_id 6
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.0001 --gpu_id 7
python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --norm_hyp 0.00005 --gpu_id 7
# python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '1gaussian' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_1gaussian' --norm_hyp 0.00002 --gpu_id 4
# python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '1gaussian' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_1gaussian' --norm_hyp 0.00001 --gpu_id 4
# python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_8gaussians' --gpu_id 0
# python realnvp_toy_experiment.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name 'k8_realnvp_checkerboard' --gpu_id 0




python realnvp_toy_experiment2.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name '[MLE+KL]k8_realnvp_8gaussians' --gpu_id 3
python realnvp_toy_experiment2.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name '[MLE+KL]k8_realnvp_checkerboard' --gpu_id 5
python realnvp_toy_experiment3.py --use_wandb 'True' --no_tensorboard --dataset '8gaussians' --num_flows 24 --learning_rate 1e-4 --experiment_name '[KL]k8_realnvp_8gaussians' --gpu_id 6
python realnvp_toy_experiment3.py --use_wandb 'True' --no_tensorboard --dataset 'checkerboard' --num_flows 24 --learning_rate 1e-4 --experiment_name '[KL]k8_realnvp_checkerboard' --gpu_id 7


