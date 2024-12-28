#!/bin/bash 

# setup environment 
conda activate mae

# ressource specs
NUM_WORKERS=8
TIME=4:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

# user name
USER_MACHINE=abizeul_euler 

# which experiment and masking ratio to run, see config/experiment
DATASET=pcmae_tiny_pc
MASK=0.2
EPOCH=800

# for imagenet, ensure to request a GPU with 80GB of RAM
if [ "$DATASET" == "pcmae_imagenet_pc" ]; then
MEM_PER_GPU=80G
fi

# Linear probe
RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_lin"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="python main.py user=abizeul_euler experiment=$DATASET masking.pc_ratio=$MASK trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# MLP probe
RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_mlp"
NAME="../$RUN_TAG"
JOB="python main.py user=abizeul_euler experiment=$DATASET masking.pc_ratio=$MASK trainer=eval_mlp checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# Fine-tuning
RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_fine"
NAME="../$RUN_TAG"
JOB="python main.py user=abizeul_euler experiment=$DATASET masking.pc_ratio=$MASK trainer=eval_fine checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# k-Nearest Neighbors
RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_knn"
NAME="../$RUN_TAG"
JOB="python main.py user=abizeul_euler experiment=$DATASET masking.pc_ratio=$MASK trainer=eval_knn checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"


