#!/bin/bash
#SBATCH -p gpu                           # Partition (queue) for GPUs
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-gpu=8                 # Allocate 8 CPU cores per GPU
#SBATCH -C gmem48                        # Request a specific GPU node
#SBATCH --job-name="stvg-array"          # Job name
#SBATCH --time=3-00:00:00
#SBATCH --exclude=c1-7                   # Exclude a specific node

# --- Job Array Configuration ---
# This line creates 4 tasks, with IDs 0, 1, 2, 3. 
# Each task will process a different chunk of the data.
# The output logs for each task will be saved to results/JOBID_TASKID.out
# %A is the master Job ID, %a is the array task ID.
#SBATCH --array=0-4
#SBATCH --output=results/%A_%a.out

# --- Script Parameters ---
MODEL="ferret"
DATASET="rvos"
# TASK="referral"
CHUNK_SIZE=200 # Number of video entries each job task will process

# --- Dynamic Calculation for this Task ---
# Calculate the starting index for this specific task based on its array ID
# Task 0: START_INDEX = 0 * 500 = 0
# Task 1: START_INDEX = 1 * 500 = 500
# etc.
START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

# --- Dynamic File Naming ---
# Create a unique output file for each task in the job array
# LOGFILE="results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODEL}_${DATASET}_${TASK}_preds.json"
LOGFILE="results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODEL}_${DATASET}_preds.json"

echo "--- SLURM JOB ARRAY TASK ---"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing ${CHUNK_SIZE} entries starting from index ${START_INDEX}"
echo "Output file: ${LOGFILE}"
echo "--------------------------"

# Print GPU and system info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load modules
module load anaconda3/2019.03-1
module load cuda/12.1

# Activate Virtual Environment
source activate stvg

# --- ntfy Notification Setup ---
NTFY_TOPIC="stvg-crcv-cluster-alerts"

finish() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        MESSAGE="✅ Job Array Task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} finished successfully."
    else
        MESSAGE="❌ Job Array Task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} failed with exit code $EXIT_CODE."
    fi
    curl -H "Title: Slurm Job Update" -d "$MESSAGE" "https://ntfy.sh/$NTFY_TOPIC"
}
trap finish EXIT
# --- End of Notification Setup ---

# --- Run the Evaluation Script ---
# Use the dynamically calculated variables to run the python script
/home/aparcedo/.conda/envs/stvg/bin/python run_eval.py \
    --dataset $DATASET \
    --model $MODEL \
    --output_path $LOGFILE \
    --device cuda \
    --entry_index $START_INDEX \
    --max_iters $CHUNK_SIZE \
    # --task_type $TASK \

echo "Task ${SLURM_ARRAY_TASK_ID} is complete."