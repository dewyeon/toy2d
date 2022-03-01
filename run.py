import subprocess
import argparse
parser=argparse.ArgumentParser(description="run")
parser.add_argument("--n", type=int)
args = parser.parse_args()
n = args.n

if args.n == 0:
    for ww in [1e-10, 1e-5]:
        for k in [4, 8]:
            for it in [100000, 200000]:
                for lr in [1e-4, 2e-4]:
                    subprocess.call(f"python toy_experiment.py --dataset 8gaussians -K {k} --weight_decay {ww} --learning_rate {lr} --experiment_name {'k'+str(k)+'_ww'+str(ww)} --gpu {n}", shell=True)

if args.n == 1:
    for ww in [1e-10, 1e-5]:
        for k in [4, 8]:
            for it in [100000, 200000]:
                for lr in [1e-4, 2e-4]:
                    subprocess.call(f"python toy_experiment.py --dataset checkerboard -K {k} --weight_decay {ww} --learning_rate {lr} --experiment_name {'k'+str(k)+'_ww'+str(ww)} --gpu {n}", shell=True)

if args.n == 2:
    for ww in [1e-10, 1e-5]:
        for k in [4, 8]:
            for it in [100000, 200000]:
                for lr in [1e-4, 2e-4]:
                    subprocess.call(f"python toy_experiment.py --dataset 2spirals -K {k} --weight_decay {ww} --learning_rate {lr} --experiment_name {'k'+str(k)+'_ww'+str(ww)} --gpu {n}", shell=True)

if args.n == 3:
    for ww in [1e-10, 1e-5]:
        for k in [4, 8]:
            for it in [100000, 200000]:
                for lr in [1e-4, 2e-4]:
                    subprocess.call(f"python toy_experiment.py --dataset moons -K {k} --weight_decay {ww} --learning_rate {lr} --experiment_name {'k'+str(k)+'_ww'+str(ww)} --gpu {n}", shell=True)

if args.n == 4:
    for ww in [1e-10, 1e-5]:
        for k in [4, 8]:
            for it in [100000, 200000]:
                for lr in [1e-4, 2e-4]:
                    subprocess.call(f"python toy_experiment.py --dataset pinwheel --batch_size 255 --plot_resolution 255 -K {k} --weight_decay {ww} --learning_rate {lr} --experiment_name {'k'+str(k)+'_ww'+str(ww)} --gpu {n}", shell=True)

if args.n == 5:
    for cb in [4, 8]:
        for dataset in ['moons', 'pinwheel', 'checkerboard', '2spirals']:
            subprocess.call(f"python cflow_toy_exp.py -cb {cb} --dataset {dataset} --learning_rate 1e-4 --experiment_name {'cflow_cb'+str(cb)} --gpu {n}", shell=True)
