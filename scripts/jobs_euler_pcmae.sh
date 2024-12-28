#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=12G
MEM_PER_GPU=24G

# DATASETs=(pcmae_blood_tvb pcmae_blood_bvt pcmae_cifar10_tvb pcmae_cifar10_bvt pcmae_derma_tvb pcmae_derma_bvt)
# MASKs=(0.6 0.7 0.8 0.85 0.9)
# DATASETs=(pcmae_cifar10_pc pcmae_tiny_pc pcmae_blood_pc pcmae_derma_pc pcmae_path_pc)
DATASETs=(pcmae_tiny_pc)
MASKs=(0.1 0.2 0.3 0.4 0.5)

for DATASET in "${DATASETs[@]}"
do
for MASK in "${MASKs[@]}"
do
RUN_TAG=""$DATASET"_pc_pc_"$MASK"_lossA"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pc_ratio=$MASK run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:24g --wrap="nvidia-smi;$JOB"
done; done;

