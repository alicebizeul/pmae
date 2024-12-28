#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=2:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

##### MAE Baselines patch 8
# DATASETs=(pcmae_tiny_pcsampling pcmae_tiny_ratiosampling pcmae_cifar10_pcsampling pcmae_cifar10_ratiosampling pcmae_blood_pcsampling pcmae_blood_ratiosampling pcmae_path_pcsampling pcmae_path_ratiosampling pcmae_derma_pcsampling pcmae_derma_ratiosampling)
DATASETs=(pcmae_path_pcsampling_rest pcmae_tiny_pcsampling_rest pcmae_blood_pcsampling_rest pcmae_derma_pcsampling_rest pcmae_cifar10_pcsampling_rest)
MODEL="vit-b"
# EPOCHs=(100 200 300 400 500 600 700 800)
EPOCHs=(800)

for DATASET in "${DATASETs[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
RUN_TAG=""$DATASET"_eval_"$EPOCH"_model_"$MODEL"_lin"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/anaconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH model=$MODEL run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

RUN_TAG=""$DATASET"_eval_"$EPOCH"_model_"$MODEL"_knn"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/anaconda3/envs/mae/bin/python main.py user=abizeul_eulerexperiment=$DATASET trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH model=$MODEL run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;