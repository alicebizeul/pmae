#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=24G

DATASETs=(pcmae_tiny_pcsampling_rest pcmae_cifar10_pcsampling_rest pcmae_blood_pcsampling_rest pcmae_path_pcsampling_rest pcmae_derma_pcsampling_rest )
MODEL="vit-b"
for DATASET in "${DATASETs[@]}"
do
RUN_TAG=""$DATASET"_model_"$MODEL"_lossA"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/anaconda3/envs/mae/bin/python main.py user=abizeul_euler run_tag=$RUN_TAG experiment=$DATASET model=$MODEL"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done;
