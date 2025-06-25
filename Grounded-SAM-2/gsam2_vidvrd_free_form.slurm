#!/bin/bash
#SBATCH -p gpu                          # Partition (queue) for GPUs
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=8                 # Allocate 6 CPU cores per GPU
#SBATCH -C "gmem32"                      # Request a specific GPU node
#SBATCH --job-name="gsam2_vidvrd_eval"             # Job name
#SBATCH --gres-flags=enforce-binding     # Enforce resource binding
#SBATCH --output=vidvrd_eval_result/free_form/log/log.out  # Output log
#SBATCH --exclude=c2-1                   # Exclude node c2-1

# Print GPU and system info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
# load module
module load anaconda3
module load cuda/11.8
# Activate Virtual Environment
source activate gsam
# Navigate to the GroundingDINO directory
cd /home/we337236/stvg/Grounded-SAM-2
echo "running gsam2_vidvrd_free_form.py"
# Run the code
python gsam2_vidvrd_free_form.py  
echo "finished gsam2_vidvrd_free_form.py"
