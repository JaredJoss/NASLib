#!/bin/bash

# search space and datasets:
search_space=$1
dataset=$2
start_seed=$3

if [ -z "$search_space" ]
then
    echo Search space argument not provided
    exit 1
fi

if [ -z "$dataset" ]
then
    echo Dataset argument not provided
    exit 1
fi

if [ -z "$start_seed" ]
then
    start_seed=0
fi

out_dir=run
trials=10
end_seed=$(($start_seed + $trials - 1))
train_sizes=(2)
test_size=200
config_root=configs

for train_size in "${train_sizes[@]}"
do

python scripts/create_configs_xgb_correlation.py --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --config_root=$config_root --zc_names flops params snip jacov grad_norm plain epe_nas fisher grasp l2_norm synflow nwot \
    --train_size $train_size

done