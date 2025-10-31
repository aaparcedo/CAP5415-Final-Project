#!/bin/bash
#SBATCH -p gpu                           # Partition (queue) for GPUs
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=8                 # Allocate 6 CPU cores per GPU
#SBATCH -C gmem48                        # Request a specific GPU node
#SBATCH --job-name="stvg"                # Job name
#SBATCH --time=3-00:00:00
#SBATCH --output=results/%j.out          # Output log
#SBATCH --exclude=c1-7                   # Exclude node


MODEL="cogvlm"
DATASET="rvos"
# TASK="referral"

# LOGFILE="results/${SLURM_JOB_ID}_${MODEL}_${DATASET}_${TASK}_part4of4_predictions.json"
LOGFILE="results/${SLURM_JOB_ID}_${MODEL}_${DATASET}_preds.json"

echo $LOGFILE

# Print GPU and system info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
# load module
module load anaconda3/2019.03-1
module load cuda/12.1
# Activate Virtual Environment
source activate stvg
conda env list

# --- ntfy Notification Setup ---
NTFY_TOPIC="stvg-crcv-cluster-alerts" # Change to your secret topic

finish() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        MESSAGE="✅ Job '$LOGFILE' ($SLURM_JOB_ID) finished successfully."
    else
        MESSAGE="❌ Job '$LOGFILE' ($SLURM_JOB_ID) failed with exit code $EXIT_CODE."
    fi
    wget \
      --header="Title: Slurm Job Update" \
      --post-data="$MESSAGE" \
      "https://ntfy.sh/$NTFY_TOPIC" \
      -O /dev/null # Suppress output to terminal
}
trap finish EXIT
# --- End of Notification Setup ---

# Run the code
# Referral vs Freeform Tasks
# /home/aparcedo/.conda/envs/stvg/bin/python run_eval.py --dataset $DATASET --model $MODEL --task_type $TASK --output_path $LOGFILE --device cuda

# Mevis and Refer-Youtube-VOS
/home/aparcedo/.conda/envs/stvg/bin/python run_eval.py --dataset $DATASET --model $MODEL --output_path $LOGFILE --device cuda 

# 
# /home/aparcedo/.conda/envs/stvg/bin/python run_eval.py --dataset vidstg --model shikra --task_type referral --output_path results/vidstg_referral_debug.json --device cuda --entry_index 0 --frame_step 15 --max_iters 2

