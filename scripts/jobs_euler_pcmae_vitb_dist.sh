#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

TIME=120:00:00
NUM_WORKERS=8
MEM_PER_GPU=24G
NUM_GPU=4
MEM_PER_CPU=$((2 * NUM_GPU))G

DATASETs=(pcmae_imagenet_pc)
MASKs=(0.2 0.3)
MODEL="vit-b"
for DATASET in "${DATASETs[@]}"
do
for MASK in "${MASKs[@]}"
do

if [ "$DATASET" == "pcmae_imagenet_pc" ]; then
MEM_PER_GPU=80G
fi

RUN_TAG=""$DATASET"_pc_pc_"$MASK"_model_"$MODEL"_lossA"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pc_ratio=$MASK run_tag=$RUN_TAG model=$MODEL trainer.devices=$NUM_GPU"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus="$NUM_GPU" --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done;

