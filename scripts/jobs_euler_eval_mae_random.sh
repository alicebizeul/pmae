#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=2:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

##### MAE Baselines patch 8
DATASETs=(mae_sampling_tiny mae_sampling_path mae_sampling_cifar10 mae_sampling_derma mae_sampling_blood)
# DATASETs=(pcmae_path_pcsampling_rest pcmae_tiny_pcsampling_rest pcmae_blood_pcsampling_rest pcmae_derma_pcsampling_rest pcmae_cifar10_pcsampling_rest)

EPOCHs=(800)
# EPOCHs=(800)

for DATASET in "${DATASETs[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
# RUN_TAG=""$DATASET"_eval_"$EPOCH"_lin"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# RUN_TAG=""$DATASET"_eval_"$EPOCH"_mlp"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET trainer=eval_mlp checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_fine"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET trainer=eval_fine checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# RUN_TAG=""$DATASET"_eval_"$EPOCH"_knn"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler  experiment=$DATASET trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;
