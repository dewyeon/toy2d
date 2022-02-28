# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
exp_name=toy_matching
seed=123
resolution=500  # resolution of plot images
logging=1000
dataset=u6
batch_size=16

num_flows=1
regularization_rate=0.6

iters_per_component=50000
num_components=2
num_steps=$((iters_per_component * num_components * 2))  # can make 2 passes training each component

python -m toy_experiment --dataset ${dataset} \
       --experiment_name ${exp_name} \
       --print_log \
       --no_cuda \
       --num_workers 1 \
       --plot_resolution ${resolution} \
       --num_steps ${num_steps} \
       --iters_per_component ${iters_per_component} \
       --learning_rate 0.005 \
       --max_grad_norm 0.0 \
       --no_lr_schedule \
       --no_annealing \
       --flow boosted \
       --rho_init uniform \
       --component_type affine \
       --num_components ${num_components} \
       --regularization_rate ${regularization_rate} \
       --num_flows ${num_flows} \
       --z_size ${z_size} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --rho_iters 0 \
       --log_interval ${iters_per_component} \
       --plot_interval ${iters_per_component} ;

echo "Job complete"

    
