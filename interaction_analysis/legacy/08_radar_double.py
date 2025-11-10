# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm 
from pathlib import Path 

# --- Import constants ---
# Make sure your constants.py file has these
from constants import (
    ST_COARSE_CATEGORIES_MAP, ENTITY_COARSE_CATEGORIES_MAP,
    ST_COARSE_COLOR_MAP, ENTITY_COARSE_COLOR_MAP,
    UNIVERSAL_FONTSIZE
)
# ---

# ---
# ---  SINGLE CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv",
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv",
    
    # --- This is the new combined output file ---
    "output_savename": "radar_charts_combined.svg", 

    # --- Data Dirs ---
    "dalton_results_dir": "/home/aparcedo/IASEB/results/all_final_results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/results/all_final_results/final_aka_on_paper/detection",
    "anirudh_results_dir": "/home/aparcedo/IASEB/results/all_final_results/STVG_results_anirudh"
}
# ---
# --- END CONTROL PANEL ---
# ---


# %%
# --- 1. Data Loading Function ---
def load_and_process_data(data_path, coarse_categories_map, control_panel_dirs):
    """
    Loads and processes all result files for a specific mode.
    """
    print(f"Loading data from {data_path}")
    classification_data = pd.read_csv(data_path)
    cls_data = dict(zip(
                [
                caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                    for caption in classification_data["caption"]],
                classification_data["category"]
                ))
    
    processed_records = []
    
    # READ DALTON RESULTS
    print("Processing Dalton results...")
    BASE_DIR = control_panel_dirs["dalton_results_dir"]
    filenames = os.listdir(BASE_DIR)
    for fn in filenames:
        try: data = json.load(open(os.path.join(BASE_DIR, fn), 'r'))
        except: continue
        for sample_dict in tqdm(data, desc=f"Dalton: {fn}", leave=False):
            caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
            category_str = cls_data.get(caption)
            if not category_str: continue
            processed_records.append({"model": fn.split("_")[0], "dataset": fn.split("_")[1], "caption": caption, "coarse_category": coarse_categories_map[int(category_str[0])], "fine_category": category_str, "mvIoU": sample_dict["mvIoU"]})

    # READ ALEJANDRO RESULTS
    print("Processing Alejandro results...")
    BASE_DIR = control_panel_dirs["alejandro_results_dir"]
    filenames = os.listdir(BASE_DIR)
    for fn in filenames:
        if ".json" not in fn: continue
        try: data = json.load(open(os.path.join(BASE_DIR, fn), 'r'))
        except: continue
        for sample_dict in tqdm(data["results"], desc=f"Alejandro: {fn}", leave=False):
            if "rvos" in fn or "mevis" in fn: sample_dict = sample_dict["entry"]
            caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
            category_str = cls_data.get(caption)
            if not category_str: continue
            processed_records.append({"model": fn.split("_")[1], "dataset": fn.split("_")[2], "caption": caption, "coarse_category": coarse_categories_map[int(category_str[0])], "fine_category": category_str, "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))})

    # READ ANIRUDH RESULTS
    print("Processing Anirudh's results...")
    BASE_DIR = Path(control_panel_dirs["anirudh_results_dir"])
    json_files = list(BASE_DIR.rglob('*.json'))
    for file_path in tqdm(json_files, desc="Processing files"):
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
            processed_records.append({"model": model, "dataset": dataset, "caption": caption, "coarse_category": coarse_categories_map[int(category_str[0])], "fine_category": category_str, "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))})
    
    return pd.DataFrame(processed_records)


# %%
# --- 2. Category Cleaning Functions ---

# --- ST-specific Roll-up and Short Names ---
ROLLUP_MAP_ST = {
    "1.2.1 Supportive Contact": "1.2 Contact",
    "1.2.2 Manipulative Contact": "1.2 Contact",
    "1.2.3 Social/Affectionate Contact": "1.2 Contact",
    "1.3.1 Gaze": "1.3 Perceptual & Indicative Relationships",
    "1.3.2 Indicative Gesture": "1.3 Perceptual & Indicative Relationships",
    "3.1.1 Approach & Depart": "3.1 Relative Motion",
    "3.1.2 Passing & Crossing": "3.1 Relative Motion",
    "3.1.2 Following & Leading": "3.1 Relative Motion", 
    "3.1.3 Following & Leading": "3.1 Relative Motion",
    "3.2 Object State Change": "2.2 Object State Change", 
    "3.3 Composite Action Sequences": "3.4 Composite Action Sequences", 
    "3.1.3 Instantaneous Motion & Impact": "3.3 Instantaneous Motion & Impact" 
}
SHORT_NAME_MAP_ST = {
    "1.1 Relative Position": "1.1 Position",
    "1.2 Contact": "1.2 Contact",
    "1.3 Perceptual & Indicative Relationships": "1.3 Perceptual",
    "1.4 Communicative Acts": "1.4 Communication",
    "2.1 Actor State Change": "2.1 Actor State",
    "2.2 Object State Change": "2.2 Object State",
    "2.3 Sequential Actions": "2.3 Sequences",
    "2.4 Durational States & Non-Actions": "2.4 Durational",
    "3.0 ST Composite": "3.0 Composite",
    "3.1 Relative Motion": "3.1 Rel. Motion",
    "3.2 Object Transference": "3.2 Transference",
    "3.3 Instantaneous Motion & Impact": "3.3 Motion/Impact",
    "3.4 Composite Action Sequences": "3.4 Sequences"
}

def get_plot_category_st(fine_category_str):
    rolled_up_name = ROLLUP_MAP_ST.get(fine_category_str, fine_category_str)
    return SHORT_NAME_MAP_ST.get(rolled_up_name, rolled_up_name)

# --- Entity-specific Naming (simpler, just uses coarse name) ---
def get_plot_category_entity(coarse_category_str):
    # This just splits the name for better layout
    return coarse_category_str.replace('-', '-\n')


# %%
# --- 3. Refactored Plotting *Helper* Function ---
# This is based on the *style* of your script (horizontal text)
# but fixed to prevent overlapping and squishing.

def _draw_radar_on_ax(ax, df, coarse_color_map, 
                      category_col, 
                      coarse_category_col='coarse_category',
                      universal_fontsize=7):
    """
    Internal helper function to draw a radar chart on a *given* ax.
    Uses horizontal text labels.
    """
    plot_df = df.dropna(subset=[category_col, coarse_category_col])
    if plot_df.empty:
        ax.text(0.5, 0.5, f"No data for\n{category_col}", ha='center', va='center')
        return

    categories = plot_df[category_col].unique(); categories.sort()
    models = plot_df['model'].unique(); models.sort()
    
    if len(categories) < 3: return

    pivot_df = plot_df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0)

    coarse_map = pd.Series(plot_df[coarse_category_col].values, index=plot_df[category_col]).to_dict()
    sorted_categories = list(pivot_df.columns)
    try: sorted_coarse_categories = [coarse_map[cat] for cat in sorted_categories]
    except KeyError as e: print(f"Error: {e}"); return

    max_iou = pivot_df.values.max(); data_max = max_iou + 0.05
    if data_max == 0.05: data_max = 1.0

    num_vars = len(categories)
    angles_plot = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_plot += angles_plot[:1] 
    angles_labels = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_rlabel_position(0)
    ax.set_xticks(angles_labels)
    ax.set_xticklabels([]) 

    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist(); values += values[:1]
        color_map = plt.cm.get_cmap('tab20'); color = color_map(i % color_map.N)
        ax.plot(angles_plot, values, color=color, linewidth=1, label=model, zorder=2)
        ax.fill(angles_plot, values, color=color, alpha=0.25, zorder=2)

    arc_inner_radius = data_max
    arc_thickness_ratio = 0.08
    arc_outer_radius = data_max * (1 + arc_thickness_ratio)
    segment_width = 2 * np.pi / num_vars
    start_angles = angles_labels

    for i in range(num_vars):
        coarse_cat = sorted_coarse_categories[i]
        color = coarse_color_map.get(coarse_cat, '#808080')
        ax.bar(x=start_angles[i], height=arc_outer_radius - arc_inner_radius,
               width=segment_width, bottom=arc_inner_radius,
               color=color, alpha=0.7, zorder=1, align='edge')

    ax.spines['polar'].set_visible(False)
    
    # --- This section draws the labels, as in your script ---
    # We set the radius *outside* the ring to prevent overlap
    label_radius = arc_outer_radius * 1.05 
    
    for i, category in enumerate(categories):
        angle_rad = angles_labels[i] 
        ax.text(angle_rad, 
                label_radius, # Use the radius to set distance
                category,
                fontsize=universal_fontsize,
                fontweight='normal', # 'light' is hard to read in a paper
                rotation=0,  # Keep all text horizontal
                ha='center',   
                va='center',  
                color='black',
                zorder=3)
    
    ax.set_yticks(np.arange(0, data_max, data_max / 5))
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, data_max, data_max / 5)], 
                       color="grey", size=universal_fontsize - 1)
    
    # We set the Y-Limit based on the labels to prevent clipping
    final_chart_max = label_radius * 1.15
    ax.set_ylim(0, final_chart_max)


# %%
# --- 4. Main Execution Block ---

print("--- Loading Spatio-Temporal (ST) Data ---")
df_st = load_and_process_data(
    data_path=CONTROL_PANEL['st_data_path'],
    coarse_categories_map=ST_COARSE_CATEGORIES_MAP,
    control_panel_dirs=CONTROL_PANEL
)
# Apply the ST roll-up
df_st['plot_category'] = df_st['fine_category'].apply(get_plot_category_st)
print(f"\nLoaded {len(df_st)} ST records.")


print("\n--- Loading Entity Data ---")
df_entity = load_and_process_data(
    data_path=CONTROL_PANEL['entity_data_path'],
    coarse_categories_map=ENTITY_COARSE_CATEGORIES_MAP,
    control_panel_dirs=CONTROL_PANEL
)
# For entity, we just plot by the coarse category, with a line break
df_entity['plot_category'] = df_entity['coarse_category'].apply(get_plot_category_entity)
print(f"\nLoaded {len(df_entity)} Entity records.")


print("\n--- Creating Combined Plot ---")

# --- Use a CVPR 2-column width (7in) and a 1:1 aspect for each subplot ---
fig, (ax1, ax2) = plt.subplots(
    1, 2, 
    figsize=(7, 3.45), # 7in wide, 3.45in tall
    subplot_kw=dict(polar=True)
)

# Set a font size that works for the small figure
CHART_FONTSIZE = 7

# --- Draw the ST Chart (Left) ---
_draw_radar_on_ax(
    ax=ax1,
    df=df_st,
    coarse_color_map=ST_COARSE_COLOR_MAP,
    category_col='plot_category', # Use the new rolled-up column
    coarse_category_col='coarse_category',
    universal_fontsize=CHART_FONTSIZE
)
ax1.set_title("Spatio-Temporal", fontsize=CHART_FONTSIZE+1, y=1.20)


# --- Draw the Entity Chart (Right) ---
_draw_radar_on_ax(
    ax=ax2,
    df=df_entity,
    coarse_color_map=ENTITY_COARSE_COLOR_MAP,
    category_col='plot_category', # Use the new line-broken coarse name
    coarse_category_col='coarse_category',
    universal_fontsize=CHART_FONTSIZE
)
ax2.set_title("Entity-Based", fontsize=CHART_FONTSIZE+1, y=1.20)


# --- Create the Shared Legend ---
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels, 
    loc='center left', 
    bbox_to_anchor=(0.95, 0.5), # Position it
    fontsize=CHART_FONTSIZE
)

# Adjust layout to prevent overlap and make room for legend
fig.tight_layout(rect=[0, 0, 0.88, 1]) # Leave 12% on the right for legend

# --- Save the final figure ---
save_path = CONTROL_PANEL['output_savename']
fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
plt.show()

print(f"Combined radar chart saved to {save_path}")
print("Done.")

# %%