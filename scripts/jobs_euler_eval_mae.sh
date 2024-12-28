#!/bin/bash 

# setup environment 
module load stack/2024-06 python/3.11.6
conda activate mae

cd /cluster/home/abizeul/mae

NUM_WORKERS=8
TIME=4:00:00
MEM_PER_CPU=2G
MEM_PER_GPU=12G

DATASETs=(mae_tiny mae_cifar10 mae_blood mae_derma mae_path)
# DATASETs=(mae_blood mae_derma)
# EPOCHs=(100 200 300 400 500 600 700 800)
EPOCHs=(800)
# MASKs=(0.75 0.5 0.6 0.8 0.9)
MASKs=(0.75)
for DATASET in "${DATASETs[@]}"
do
for EPOCH in "${EPOCHs[@]}"
do
for MASK in "${MASKs[@]}"
do


# RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_lin"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_lin checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_transfert"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_transfert checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"


# RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_mlp"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_mlp checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_fine"
NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_fine checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"

# RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_transfert_fine"
# NAME="/cluster/home/abizeul/mae/output_log/$RUN_TAG"
# JOB="/cluster/home/abizeul/miniconda3/envs/mae/bin/python main.py user=abizeul_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_transfert_fine checkpoint=pretrained checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"


# RUN_TAG=""$DATASET"_pc_"$MASK"_eval_"$EPOCH"_model_"$MODEL"_knn"
# NAME="/cluster/home/callen/projects/mae/output_log/$RUN_TAG"
# JOB="python main.py user=callen_euler experiment=$DATASET masking.pixel_ratio=$MASK trainer=eval_knn checkpoint=pretrained model=$MODEL checkpoint.epoch=$EPOCH run_tag=$RUN_TAG"
# sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
done; done; done;
