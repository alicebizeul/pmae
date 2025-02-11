#!/bin/bash 

# setup environment 
conda activate mae

# ressource specs
NUM_WORKERS=8
TIME=120:00:00
MEM_PER_CPU=12G
MEM_PER_GPU=24G

# user name
USER_MACHINE=abizeul_euler 

# which experiment and masking ratio to run, see config/experiment
EXPERIMENT=pcmae_cifar10_pc
MASK=0.2

# for imagenet, ensure to request a GPU with 80GB of RAM
if [ "$EXPERIMENT" == "pcmae_imagenet_pc" ]; then
MEM_PER_GPU=80G
fi

RUN_TAG=""$EXPERIMENT"_mask_"$MASK""
NAME="../$RUN_TAG"
JOB="python main.py user=$USER_MACHINE experiment=$EXPERIMENT masking.pc_ratio=$MASK run_tag=$RUN_TAG"
sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:24g --wrap="nvidia-smi;$JOB"

#module_config.norm_pix_loss=True