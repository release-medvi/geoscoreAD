#!/bin/bash

GPUS=(1 1)   # use free GPUs

datasets=("liver" "brats2021" "RESC" )
methods=( "adaclip" "anomalyclip")
shots=(1 5)

mkdir -p logs
gpu_index=0

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for k in "${shots[@]}"; do

            GPU=${GPUS[$gpu_index]}
            gpu_index=$(( (gpu_index + 1) % 2 ))

            CONFIG=config/${dataset}_${method}.yaml
            LOG=logs/${method}_${dataset}_shot_${k}.txt

            echo "Running $method | $dataset | few-shot $k on GPU $GPU"

            CUDA_VISIBLE_DEVICES=$GPU python main.py \
                --config $CONFIG \
                --num_few_shot $k \
                &> $LOG
        done
    done
done
