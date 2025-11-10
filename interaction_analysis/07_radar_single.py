# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm 
from pathlib import Path 

# --- Import constants ---
from constants import (
    ST_COARSE_CATEGORIES_MAP, ENTITY_COARSE_CATEGORIES_MAP,
    ST_COARSE_COLOR_MAP, ENTITY_COARSE_COLOR_MAP,
    UNIVERSAL_FONTSIZE # <-- IMPORT THE FONTSIZE
)
# ---


# ---
# ---  SINGLE CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- Mode Switch ---
    "DATA_MODE": "st",  

    # --- File Paths ---
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/legacy/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv", # GOING OFF THIS FILE PATH - THIS VERSION OF THE SCRIPT DOES NOT INCLUDE ANY MEVIS/RVOS RESULTS
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/legacy/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv",
    
    # --- Output Filenames ---
    "coarse_chart_output": "radar_chart_coarse.svg",
    "fine_chart_output": "entity_all_radar_chart_fine.svg",

    # --- Data Directories ---
    "dalton_results_dir": "/home/aparcedo/IASEB/results/all_final_results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/results/all_final_results/final_aka_on_paper_alejandro/detection",
    "wen_results_dir": "/home/aparcedo/IASEB/results/all_final_results/stvg_output_bbox_wen",
    "anirudh_results_dir": "/home/aparcedo/IASEB/results/all_final_results/STVG_results_anirudh"
}
# ---
# --- END CONTROL PANEL ---
# ---

# %%
# --- 1. LOAD DATA BASED ON MODE ---
# (Unchanged, correctly sets up coarse_categories_map and coarse_color_map_for_plot)
mode = CONTROL_PANEL["DATA_MODE"]
if mode == "st":
    data_path = CONTROL_PANEL["st_data_path"]
    coarse_categories_map = ST_COARSE_CATEGORIES_MAP
    coarse_color_map_for_plot = ST_COARSE_COLOR_MAP
    print("Running in 'spatiotemporal' (st) mode.")
elif mode == "entity":
    data_path = CONTROL_PANEL["entity_data_path"]
    coarse_categories_map = ENTITY_COARSE_CATEGORIES_MAP
    coarse_color_map_for_plot = ENTITY_COARSE_COLOR_MAP
    print("Running in 'entity' mode.")
else:
    raise ValueError(f"DATA_MODE in CONTROL_PANEL must be 'st' or 'entity', not '{mode}'")
print(f"Loading data from {data_path}")
classification_data = pd.read_csv(data_path)
cls_data = dict(zip(
            [
            caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                for caption in classification_data["caption"]],
            classification_data["category"]
            ))
processed_records = [] 


# %%
# --- 2. PROCESS RESULTS ---
# (Unchanged, processing logic is correct)
# READ DALTON RESULTS
print("Processing Dalton results...")
BASE_DIR = CONTROL_PANEL["dalton_results_dir"]
filenames = os.listdir(BASE_DIR)
for fn in filenames:
    if "referral" in fn:
        print(f'skipping: {fn}')
        continue
    try: data = json.load(open(os.path.join(BASE_DIR, fn), 'r'))
    except: continue
    for sample_dict in tqdm(data, desc=f"Dalton: {fn}", leave=False):
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        category_str = cls_data.get(caption)
        if not category_str: continue
        processed_records.append({"model": fn.split("_")[0], "dataset": fn.split("_")[1], "caption": caption, "coarse_category": coarse_categories_map[int(category_str[0])], "fine_category": category_str, "mvIoU": sample_dict["mvIoU"]})
# READ ALEJANDRO RESULTS
print("Processing Alejandro results...")
BASE_DIR = CONTROL_PANEL["alejandro_results_dir"]
filenames = os.listdir(BASE_DIR)
cnt = 0
for fn in filenames:
    if "referral" in fn: 
        print(f'skipping: {fn}')
        continue
    if ".json" not in fn: continue
    try: data = json.load(open(os.path.join(BASE_DIR, fn), 'r'))
    except: continue
    for sample_dict in tqdm(data["results"], desc=f"Alejandro: {fn}", leave=False):
        if "rvos" in fn or "mevis" in fn: sample_dict = sample_dict["entry"]
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        category_str = cls_data.get(caption)
        if not category_str: continue
        else: cnt += 1
        processed_records.append({"model": fn.split("_")[1], "dataset": fn.split("_")[2], "caption": caption, "coarse_category": coarse_categories_map[int(category_str[0])], "fine_category": category_str, "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))})
# READ ANIRUDH RESULTS
print("Processing Anirudh's results...")
BASE_DIR = Path(CONTROL_PANEL["anirudh_results_dir"])
json_files = list(BASE_DIR.rglob('*.json'))
cnt = 0
for file_path in tqdm(json_files, desc="Processing files"):
    if "referral" in fn: 
        print(f'skipping: {file_path}')
        continue
    try:
        relative_parts = file_path.relative_to(BASE_DIR).parts
        if len(relative_parts) < 3: continue
        dataset = relative_parts[0]; model = file_path.stem
    except ValueError: continue
    data = json.load(open(file_path, 'r'))
    for sample_dict in data["results"]:
        caption = sample_dict["entry"]["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        category_str = cls_data.get(caption)
        if not category_str: continue
        else: cnt += 1
        processed_records.append({"model": model, "dataset": dataset, "caption": caption, "coarse_category": coarse_categories_map[int(category_str[0])], "fine_category": category_str, "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))})
print(f"Processed {cnt} records from Anirudh's results.")

# %%
df = pd.DataFrame(processed_records)
print(f'Length of data frame: {len(df)}')

# %%
# --- 1. Define the Renaming Map ---
# This map will shorten all category names to 1-2 words.
CATEGORY_NAME_MAP = {
    # --- Level 1 ---
    "1.1 Relative Position": "1.1 Position",
    "1.2.1 Supportive Contact": "1.2 Support",
    "1.2.2 Manipulative Contact": "1.2 Manipulation",
    "1.2.3 Social/Affectionate Contact": "1.2 Social Contact",
    "1.3.1 Gaze": "1.3 Gaze",
    "1.3.2 Indicative Gesture": "1.3 Gesture",
    "1.4 Communicative Acts": "1.4 Communication",
    
    # --- Level 2 ---
    "2.1 Actor State Change": "2.1 Actor State",
    "2.2 Object State Change": "2.2 Object State",
    "2.3 Sequential Actions": "2.3 Sequences",
    "2.4 Durational States & Non-Actions": "2.4 Durational",
    
    # --- Level 3 ---
    "3.0 ST Composite": "3.0 Composite",
    "3.1.1 Approach & Depart": "3.1 Approach/Depart",
    "3.1.2 Passing & Crossing": "3.1 Passing/Crossing",
    "3.1.2 Following & Leading": "3.1 Following/Leading", # Handles duplicate number
    "3.1.3 Following & Leading": "3.1 Following/Leading", # Handles duplicate name
    "3.1.3 Instantaneous Motion & Impact": "3.1 Motion/Impact", # Handles duplicate name
    "3.2 Object Transference": "3.2 Transference",
    "3.2 Object State Change": "3.2 Object State",
    "3.3 Instantaneous Motion & Impact": "3.3 Motion/Impact",
    "3.3 Composite Action Sequences": "3.3 Action Sequences",
    "3.4 Composite Action Sequences": "3.4 Action Sequences"
}


def get_plot_category(fine_category_str):
    """
    Truncates a category string like '3.1.1 Name' to its L2 string, '3.1 Name',
    and then applies a short 1-2 word alias from the map.
    """
    # 1. First, check if the *exact* full string is in the map
    # (e.g., "3.1.1 Approach & Depart")
    if fine_category_str in CATEGORY_NAME_MAP:
        return CATEGORY_NAME_MAP[fine_category_str]

    # 2. If not, try "rolling up" L3s to L2s (e.g., "3.1.1 Name" -> "3.1 Name")
    parts = fine_category_str.split(' ', 1) 
    if len(parts) < 2:
        return fine_category_str # No name part, return as-is
        
    number_part = parts[0]
    name_part = parts[1]
    levels = number_part.split('.')
    
    if len(levels) > 2:
        l2_number = ".".join(levels[:2])
        l2_string = f"{l2_number} {name_part}"
        
        # 3. Check if this new "rolled up" L2 string is in the map
        if l2_string in CATEGORY_NAME_MAP:
            return CATEGORY_NAME_MAP[l2_string]
        
        # 4. If not, just return the rolled-up string
        return l2_string
    
    # 5. If it was already L1/L2 and not in the map, return it as-is
    return fine_category_str

# --- Create the new column ---
df['plot_category'] = df['fine_category'].apply(get_plot_category)

print("Created 'plot_category' column. Unique values:")
print(sorted(df['plot_category'].unique()))
# %%

# %%

def create_and_save_radar_chart(df, save_path=None, category_col='fine_category', 
                                color_by_model=True, universal_fontsize=UNIVERSAL_FONTSIZE): 
    """
    Generates and optionally saves a radar chart from a DataFrame.
    """
    if category_col not in df.columns: raise ValueError(f"Category column '{category_col}' not found in the DataFrame.")
    plot_df = df.dropna(subset=[category_col])
    if plot_df.empty: return
    categories = plot_df[category_col].unique(); categories.sort()
    models = plot_df['model'].unique(); models.sort()
    if len(categories) < 3: return
    pivot_df = plot_df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0)
    max_iou = pivot_df.values.max(); radar_max = max_iou + 0.05
    if radar_max == 0.05: radar_max = 1.0
    num_vars = len(categories); angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist(); angles += angles[:1]
    figsize = (10, 10) if num_vars < 15 else (min(30, num_vars * 0.8), min(30, num_vars * 0.8))
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist(); values += values[:1]
        color = plt.cm.get_cmap('tab10')(i % 10) if color_by_model else 'blue'
        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_rlabel_position(0); ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=universal_fontsize)
    ax.set_yticks(np.arange(0, radar_max, radar_max / 5))
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, radar_max, radar_max / 5)], color="grey", size=universal_fontsize)
    ax.set_ylim(0, radar_max)
    ax.legend(loc='upper right', bbox_to_anchor=(1.08, 0.835), fontsize=universal_fontsize)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight'); plt.close(fig)
        print(f"Radar chart saved to {save_path}")
    else: plt.show()


# %%
# --- 6. CREATE AND SAVE PLOT (With Coarse Ring) ---
# Make sure numpy is imported at the top of your script
import numpy as np
# Make sure UNIVERSAL_FONTSIZE is imported from constants.py
from constants import UNIVERSAL_FONTSIZE 
UNIVERSAL_FONTSIZE=18
def create_and_save_radar_chart_with_ring(df, 
                                          coarse_color_map, 
                                          save_path=None, 
                                          category_col='fine_category', 
                                          coarse_category_col='coarse_category',
                                          color_by_model=True,
                                          universal_fontsize=UNIVERSAL_FONTSIZE):
    """
    Generates and optionally saves a radar chart from a DataFrame,
    with an outer ring colored by coarse category.
    """
    if category_col not in df.columns: raise ValueError(f"Category column '{category_col}' not found in the DataFrame.")
    if coarse_category_col not in df.columns: raise ValueError(f"Coarse category column '{coarse_category_col}' not found in the DataFrame.")
    
    plot_df = df.dropna(subset=[category_col, coarse_category_col])
    if plot_df.empty: print(f"Skipping chart for '{category_col}': No data found after dropping NaN."); return

    categories = plot_df[category_col].unique(); categories.sort()
    
    # --- THIS IS THE CORRECTED LINE ---
    models = plot_df['model'].unique(); models.sort()
    # --- END CORRECTION ---
    
    if len(categories) < 3: print(f"Skipping chart for '{category_col}': Not enough categories (need at least 3)."); return

    pivot_df = plot_df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0)

    coarse_map = pd.Series(plot_df[coarse_category_col].values, index=plot_df[category_col]).to_dict()
    sorted_categories = list(pivot_df.columns)
    try: sorted_coarse_categories = [coarse_map[cat] for cat in sorted_categories]
    except KeyError as e: print(f"Error: Could not find coarse category for fine category '{e.key}'."); return

    max_iou = pivot_df.values.max(); data_max = max_iou + 0.05
    if data_max == 0.05: data_max = 1.0

    num_vars = len(categories)
    
    # Angles for the data lines
    angles_plot = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_plot += angles_plot[:1] 
    
    # Angles for labels and bars
    angles_labels = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    
    # Make figure larger to accommodate legend
    figsize = (12, 12) 
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist(); values += values[:1]
        color_map = plt.cm.get_cmap('tab20'); color = color_map(i % color_map.N) if color_by_model else 'blue'
        ax.plot(angles_plot, values, color=color, linewidth=2, label=model, zorder=2)
        ax.fill(angles_plot, values, color=color, alpha=0.25, zorder=2)

    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_rlabel_position(0)
    
    # --- MODIFICATION: Set tick positions but hide default labels ---
    ax.set_xticks(angles_labels)
    ax.set_xticklabels([]) # This line HIDES the default labels
    # --- END MODIFICATION ---

    
    arc_inner_radius = data_max
    arc_thickness_ratio = 0.08 
    arc_outer_radius = data_max * (1 + arc_thickness_ratio)
    
    segment_width = 2 * np.pi / num_vars
    start_angles = angles_labels

    for i in range(num_vars):
        coarse_cat = sorted_coarse_categories[i]
        color = coarse_color_map.get(coarse_cat, '#808080')
        ax.bar(
            x=start_angles[i],
            height=arc_outer_radius - arc_inner_radius,
            width=segment_width,
            bottom=arc_inner_radius,
            color=color,
            alpha=0.7,
            zorder=1,
            align='edge'
        )

    ax.spines['polar'].set_visible(False)
    
    # --- THIS IS THE NEW SECTION THAT ADDS CUSTOM LABELS ---
    
    # --- TUNE THIS VALUE ---
    # To move text *inside* the ring, use a value < 1.0
    # To move text *outside* the ring, use a value > 1.0
    label_radius = arc_inner_radius * 1.0
    
    for i, category in enumerate(categories):
        angle_rad = angles_labels[i] 
        
        ax.text(angle_rad, 
                label_radius, # Use the radius to set distance
                category,
                fontsize=universal_fontsize,
                fontweight='light',
                rotation=0,  # Keep all text horizontal
                ha='center',   
                va='center',  
                color='black',
                zorder=3)
    # --- END NEW SECTION ---
    
    ax.set_yticks(np.arange(0, data_max, data_max / 5))
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, data_max, data_max / 5)], color="grey", size=universal_fontsize)
    
    # Adjust final limit to make room for ring (text is now inside)
    final_chart_max = arc_outer_radius * 1.0
    ax.set_ylim(0, final_chart_max)
    
    # --- ADD LEGEND (MOVED OUTSIDE) ---
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=universal_fontsize)

    # Adjust layout to make room for legend
    fig.tight_layout(rect=[0, 0, 0.8, 1])

    # if save_path:
    #     fig.savefig(save_path, format='svg', bbox_inches='tight'); plt.close(fig)
    #     print(f"Radar chart saved to {save_path}")
    # else: 
    plt.show()
print(f"Creating radar chart with coarse ring for '{mode}' mode...")

create_and_save_radar_chart_with_ring(
    df, 
    coarse_color_map=coarse_color_map_for_plot, 
    save_path=CONTROL_PANEL['fine_chart_output'], 
    category_col='plot_category',
    coarse_category_col='coarse_category', 
    color_by_model=True
)

print("All charts created.")
# %%

# PLOT entity

create_and_save_radar_chart_with_ring(
    df, 
    coarse_color_map=coarse_color_map_for_plot, 
    save_path=CONTROL_PANEL['fine_chart_output'], 
    category_col='fine_category',
    coarse_category_col='coarse_category', 
    color_by_model=True
)

# %%
