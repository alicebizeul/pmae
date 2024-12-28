#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G


DATASETs=(pcmae_tiny_pcsampling pcmae_cifar10_pcsampling pcmae_blood_pcsampling pcmae_path_pcsampling pcmae_derma_pcsampling )
for DATASET in "${DATASETs[@]}" 
do
RUN_TAG=""$DATASET"_lossA"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
