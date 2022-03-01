import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n

if args.n == 1:
    for dataset in ['checkerboard', '2spirals', 'moons', 'pinwheel', '8gaussians']:
        subprocess.call(f"python realnvp_toy_experiment.py --no_tensorboard --dataset {dataset} --num_flows 4 --learning_rate 1e-4 --experiment_name {'k4_realnvp'+str(dataset)} --gpu {n}", shell=True)

if args.n == 2:
    for dataset in ['checkerboard', '2spirals', 'moons', 'pinwheel', '8gaussians']:
        subprocess.call(f"python realnvp_toy_experiment.py --no_tensorboard --dataset {dataset} --num_flows 8 --learning_rate 1e-4 --experiment_name {'k8_realnvp'+str(dataset)} --gpu {n}", shell=True)

