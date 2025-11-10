# %%
# Aggregate all the detection and results for IASEB 
import json
import pandas as pd
import os
from tqdm import tqdm 
from pathlib import Path 
from IASEB.datasets import DATASET_PATHS

CONTROL_PANEL = {
    "dalton_results_dir": "/home/aparcedo/IASEB/results/all_final_results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/results/all_final_results/final_aka_on_paper_alejandro/detection",
    "wen_results_dir": "/home/aparcedo/IASEB/results/all_final_results/stvg_output_bbox_wen",
    "anirudh_results_dir": "/home/aparcedo/IASEB/results/all_final_results/STVG_results_anirudh"
}

# %%
# %%
# PART 1: DETECTION
# Detection datasets: HC-STVG1&2, VidVRD, VidSTG, MeViS (converted), and Ref-Youtube-VOS (converted)
# Detection models: CogVLM, Ferret, Shikra, Qwen-VL, LLaVA-G, InternVL2.5, Minigpt, Sphinx2
processed_records = []  

# Use Path() for all to make them Path objects, not strings
anirudh_filepaths = list(Path(CONTROL_PANEL["anirudh_results_dir"]).rglob('*.json'))
alejandro_filepaths = [Path(CONTROL_PANEL['alejandro_results_dir']) / f for f in os.listdir(CONTROL_PANEL['alejandro_results_dir']) if f.endswith('.json')]
dalton_filepaths = [Path(CONTROL_PANEL['dalton_results_dir']) / f for f in os.listdir(CONTROL_PANEL['dalton_results_dir']) if f.endswith('.json')]
wen_filepaths = [Path(CONTROL_PANEL['wen_results_dir']) / f for f in os.listdir(CONTROL_PANEL['wen_results_dir']) if f.endswith('.json')]

# filepaths = anirudh_filepaths + alejandro_filepaths + dalton_filepaths + wen_filepaths
filepaths = anirudh_filepaths + alejandro_filepaths + dalton_filepaths + wen_filepaths

for fp in tqdm(filepaths, desc="Processing all results files"):
    
    fp_str = str(fp)
    data = json.load(open(fp, 'r'))
        
    if 'anirudh' in fp_str:
        relative_parts = fp.relative_to(CONTROL_PANEL["anirudh_results_dir"]).parts
        dataset = relative_parts[0]
        task = relative_parts[1]
        model = fp.stem
        loop_data = data.get("results", [])
            
    elif 'alejandro' in fp_str:
        model = fp.stem.split("_")[1]
        dataset = fp.stem.split("_")[2]
        task = fp.stem.split("_")[3]
        loop_data = data.get("results", [])
        
    elif 'dalton' in fp_str:

        model = fp.stem.split("_")[0]
        dataset = fp.stem.split("_")[1]
        task = fp.stem.split("_")[2]
        loop_data = data
    elif 'wen' in fp_str:
        model = fp.stem.split("_")[0]
        dataset = fp.stem.split("_")[1]
        task = fp.stem.split("_")[2]
        loop_data = data.get("results", [])
        
    # --- 2. Process all samples for this file ---
    for sample_dict in loop_data:
        
        # --- 3. Extract Caption/mvIoU (Logic is unique per source) ---
        caption_raw = None
        mvIoU = None

        if 'anirudh' in fp_str:
            # Anirudh's is always nested in 'entry'
            entry = sample_dict.get("entry", {})
            video_path = os.path.join(DATASET_PATHS[dataset]['video'], entry["video_path"])
            caption_raw = entry.get("caption")
            mvIoU = sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))
        
        elif 'alejandro' in fp_str:
            # Alejandro's is nested *only* for rvos/mevis
            if "rvos" in fp_str or "mevis" in fp_str:
                entry = sample_dict["entry"]
                caption_raw = entry["caption"]
                mvIoU = sample_dict["metrics"]["mv_iou"]
            else:
                caption_raw = sample_dict.get("caption")
                mvIoU = sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))
            video_path = sample_dict["video_path"]
        
        elif 'dalton' in fp_str:
            video_path = os.path.join(DATASET_PATHS[dataset]['video'], sample_dict["video_path"])
            caption_raw = sample_dict.get("caption")
            mvIoU = sample_dict.get("mvIoU")
        elif 'wen' in fp_str:
            video_path = os.path.join(DATASET_PATHS[dataset]['video'], sample_dict["video_path"])
            caption_raw = sample_dict["caption"]
            mvIoU = sample_dict.get("mvIoU")
        # import code; code.interact(local=locals())

        # --- 4. Normalize caption and get classifications ---
        caption = caption_raw.strip().lower().replace('.', '').replace('"', '').replace('\\', '')

        # --- 5. Build the record ---
        record = {
            "model": model,
            "dataset": dataset,
            "task": task,
            "caption": caption,
            "mvIoU": mvIoU,
        }
        processed_records.append(record)

df = pd.DataFrame(processed_records)
print(f'Length of data frame: {len(df)}')
df.head()
df.to_csv('alejandro_dalton_anirudh_table1.csv', index=False)
# %%

# Task wise performance
# Group by both 'model' AND 'task', then calculate the mean of 'mvIoU'
task_averages = df.groupby(['model', 'task'])['mvIoU'].mean().unstack()

print(task_averages)
# %%

# Combined performance (R&F) referral and freeform
# Calculate the combined (overall) average mvIoU per model
combined_avg = df.groupby('model')['mvIoU'].mean()

print(combined_avg)
# %%

# %% AGGREGATE RESULTS WITH CATEGORIES

st_cls_level0_avg = df[df['task'] == 'freeform'].groupby(['model', 'st_cls_level0'])['mvIoU'].mean().unstack()
entity_cls_level0_avg = df[df['task'] == 'freeform'].groupby(['model', 'entity_cls_level0'])['mvIoU'].mean().unstack()
# %%
