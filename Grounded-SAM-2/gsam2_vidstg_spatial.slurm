#!/bin/bash
#SBATCH -p gpu                          # Partition (queue) for GPUs
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=8                 # Allocate 6 CPU cores per GPU
#SBATCH -C "gmem32"                      # Request a specific GPU node
#SBATCH --job-name="gsam2_vidstg__eval"             # Job name
#SBATCH --gres-flags=enforce-binding     # Enforce resource binding
#SBATCH --output=vidstg_eval_result/log/log.out  # Output log
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
echo "running gsam2_vidstg_spatial.py"
# Run the code
python gsam2_vidstg_spatial.py  
echo "finished gsam2_vidstg_spatial.py"
