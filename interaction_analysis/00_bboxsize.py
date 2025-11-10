# %%
import json
import numpy as np
import cv2
import os

# VG1 -> VG2 -> VID -> STG -> MeViS -> Ref-YT-VOS


# %%
DATASET_PATHS = {
    "hcstvg1": {
        "video": "/home/c3-0/datasets/stvg/hcstvg1/v1/video",
        "referral": "/home/c3-0/datasets/stvg/preprocess_dump/hcstvg/hcstvg_pid_tubes_multi_sent_refined_v3/sentences_test.json", 
        "freeform": "/home/c3-0/datasets/stvg/hcstvg1/test_proc.json", 
    }, 
    "hcstvg2": {
        "video": "/home/c3-0/datasets/stvg/hcstvg2/videos",
        "referral": "/home/we337236/stvg/dataset/hcstvg_v2/hcstvgv2_sentences_test_gpt_modified.json", 
        "freeform": "/home/c3-0/datasets/stvg/hcstvg2/annotations/HCVG_val_proc.json", 
    }, 
    "vidvrd": {
        "referral": "/home/we337236/stvg/dataset/vidvrd/referral_final_output.json", 
        "freeform": "/home/we337236/stvg/dataset/vidvrd/free_form_final_output.json", 
    }, 
    "vidstg": {
        "referral": "/share/datasets/stvg/vidstg_annotations/vidstg_referral.json", 
        "freeform": "/home/we337236/stvg/dataset/vidstg/vidstg_pro_test_final_list.json", 
    }, 
    "mevis": {
        "video": "/share/datasets/stvg/MeViS/MeViS/valid_u/JPEGImages/JPEGImages",
        "metadata": "/share/datasets/stvg/mevis_annotations/valid_u/one_object_meta_expressions.json",
        "bbox": "/share/datasets/stvg/mevis_annotations/valid_u/one_obj_bbox_updated_format.json",
        "masks": "/share/datasets/stvg/mevis_annotations/valid_u/mask_dict.json"
    },
    "rvos": {
        "video": "/share/datasets/stvg/rvos_annotations/valid/JPEGImages",
        "masks": "/share/datasets/stvg/rvos_annotations/valid/Annotations",
        "metadata": "/share/datasets/stvg/rvos_annotations/valid/meta_expressions_challenge.json",
        "bbox": "/share/datasets/stvg/rvos_annotations/valid/rvos_bbox_annotations.json",
    }
}
# %%
filepaths = [DATASET_PATHS["hcstvg1"]["freeform"],
             DATASET_PATHS["hcstvg2"]["freeform"],
             DATASET_PATHS["vidvrd"]["freeform"],
             DATASET_PATHS["vidstg"]["freeform"],
             ]

for fp in filepaths:

    data = json.load(open(fp, 'r'))
    for entry in data:
        width = entry["width"]
        height = entry["height"]
        total_pixels = width * height
        per_frame_mask_ratio = []

        if "trajectory" in entry:
            for bbox in entry["trajectory"]: # for each box in tube; # bbox are [x, y, w, h] format
            
                bbox_pixels = bbox[2] * bbox[3]

                per_frame_mask_ratio.append(bbox_pixels / total_pixels)
        elif "bbox" in entry:
            for frame_id, bbox in entry["bbox"].items(): # for each box in tube; # bbox are [x, y, w, h] format
            
                bbox_pixels = bbox[2] * bbox[3]

                per_frame_mask_ratio.append(bbox_pixels / total_pixels)

        entry["avg_bbox_area_ratio"] = np.mean(np.array(per_frame_mask_ratio))

    output_fp = "bbox_area_ratio_" + os.path.basename(fp)

    with open(output_fp, 'w') as f:
        # Dump the entire modified 'data' object
        json.dump(data, f, indent=4)        
# %%
filepaths = [DATASET_PATHS["mevis"]["bbox"],
             DATASET_PATHS["rvos"]["bbox"]]

for fp in filepaths:

    data = json.load(open(fp, 'r'))
    for video_id, video_data in data["videos"].items():
        if 'mevis' in fp:
            video_path = os.path.join(DATASET_PATHS["mevis"]["video"], video_id)
        else:
            video_path = os.path.join(DATASET_PATHS["rvos"]["video"], video_id)
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        frame = cv2.imread(os.path.join(video_path, image_files[int(0)]))
        height, width, channels = frame.shape
        total_pixels = height * width

        for exp_id, exp_data in video_data["expressions"].items():
            per_frame_mask_ratio = []
            for frame_id, bbox in exp_data["trajectory"].items():
                bbox_pixels = bbox[2] * bbox[3]
                per_frame_mask_ratio.append(bbox_pixels / total_pixels)
            exp_data["avg_bbox_area_ratio"] = np.mean(np.array(per_frame_mask_ratio))

    output_fp = "bbox_area_ratio_" + os.path.basename(fp)

    with open(output_fp, 'w') as f:
        # Dump the entire modified 'data' object
        json.dump(data, f, indent=4)
# %%


# PLOT THE DATA WE JUST CREATED IN A HISTOGRAM
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

# ---
# ---  CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- Area Ratio Files ---
    # Put the full path to these files
    "area_ratio_files": {
        "hcstvg1": "hcstvg1_bbox_area_ratio_test_proc.json",
        "hcstvg2": "hcstvg2_bbox_area_ratio_HCVG_val_proc.json",
        "vidvrd": "vidvrd_bbox_area_ratio_free_form_final_output.json",
        "vidstg": "vidstg_bbox_area_ratio_pro_test_final_list.json"
    },
    
    # --- Output Filename ---
    "output_plot_filename": "bbox_area_ratio_distribution.svg"
}
# ---
# --- END CONTROL PANEL ---
# ---


def load_all_ratios(area_ratio_files_map):
    """
    Loads all area ratios from all files into a single flat list.
    """
    all_ratios = []
    print("Loading area ratio data...")
    
    for dataset_name, filepath in area_ratio_files_map.items():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f) # data is a list
        except FileNotFoundError:
            print(f"ERROR: Area ratio file not found. Halting: {filepath}")
            raise
        
        print(f"  Processing ratios from {dataset_name} ({filepath})")
        
        # Add all valid ratios from this file to the main list
        for i, entry in enumerate(data):
            ratio = entry.get("avg_bbox_area_ratio")
            # Ensure ratio is valid before adding
            if ratio is not None and not np.isnan(ratio):
                all_ratios.append(ratio)
            else:
                # Fail fast if any entry is incomplete, as per your previous request
                raise ValueError(f"Data integrity error in {filepath}: Entry {i} is missing a valid ratio.")
            
        print(f"    Found {len(data)} ratios.")
        
    print(f"Area ratio loading complete. Total entries: {len(all_ratios)}")
    return all_ratios

# ---
# --- Main Execution ---
# ---

# 1. Load all ratios into a single list
all_ratios_list = load_all_ratios(CONTROL_PANEL["area_ratio_files"])

if not all_ratios_list:
    print("Error: No ratios found. Cannot create plot.")
else:
    # 2. Plot the Histogram
    print("Creating histogram plot...")
    
    # Set the theme to match your reference
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(7, 5)) # Smaller figure size, good for histograms

    # Use the same color as your other plot
    plot_color = "#8da2cc"

    # Create the histogram
    sns.histplot(
        data=all_ratios_list,
        bins=20,                 # 20 bins, just like your reference
        binrange=(0.0, 1.0),     # Lock the range from 0 to 1
        color=plot_color
    )

    # 3. Set Ticks and Labels (to match reference)
    ax = plt.gca() # Get the current axis
    
    # Set the x-axis ticks
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    
    # Set labels
    ax.set_xlabel("Area ratio of each box in the image")
    ax.set_ylabel("Count")
    ax.set_title("") # No main title, as per your reference
    
    plt.tight_layout()

    # 4. Save the figure
    output_filename = CONTROL_PANEL["output_plot_filename"]
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"\nSuccess! Histogram saved to: {output_filename}")
    plt.show()
# %%
