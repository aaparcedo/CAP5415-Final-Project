#!/bin/bash
#SBATCH -p gpu                           # Partition (queue) for GPUs
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=8                 # Allocate 6 CPU cores per GPU
#SBATCH -C gmem48                        # Request a specific GPU node
#SBATCH --job-name="NUMBERS"                # Job name
#SBATCH --time=1-00:00:00
#SBATCH --output=results/%j.out          # Output log
#SBATCH --exclude=c1-7                   # Exclude node

DATASET="rvos"
LOGFILE="/home/aparcedo/IASEB/results/${SLURM_JOB_ID}_videolisa_${DATASET}_preds.json"

echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Output file: ${LOGFILE}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
module load anaconda3/2019.03-1
module load cuda/12.1
source activate videolisa
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
/home/aparcedo/.conda/envs/videolisa/bin/python videolisa_eval.py --output_path $LOGFILE --dataset ${DATASET} 