#!/bin/bash
#SBATCH -p gpu                          # Partition (queue) for GPUs
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=8                 # Allocate 6 CPU cores per GPU
#SBATCH -C "gmem32"                      # Request a specific GPU node
#SBATCH --job-name="gsam_hcstvgv1_referral_eval"             # Job name
#SBATCH --gres-flags=enforce-binding     # Enforce resource binding
#SBATCH --output=my_outputs/hcstvgv1/referral/log.out  # Output log
#SBATCH --exclude=c2-1                   # Exclude node c2-1

# Print GPU and system info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
# load module
module load anaconda3
module load cuda/11.8
# Activate Virtual Environment
source activate gsam1
# Navigate to the gsam directory
cd /home/we337236/stvg/Grounded-Segment-Anything
echo "running gsam_hcstvg.py"
# Run the code
python gsam_hcstvg.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --output_dir "my_outputs" \
  --device "cuda"\
  --dataset_version 1\
  --caption_type "referral"
echo "finished gsam_hcstvg.py"
