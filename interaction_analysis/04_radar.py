# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm 

# ---
# ---  SINGLE CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- Mode Switch ---
    # Change this to 'st' for spatiotemporal or 'entity' for entity classes
    "DATA_MODE": "entity",  

    # --- File Paths ---
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv",
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv",
    
    # --- Output Filenames ---
    # These will be used for the final plots
    "coarse_chart_output": "radar_chart_coarse.svg",
    "fine_chart_output": "radar_chart_fine.svg",
    
    # --- Data Directories ---
    "dalton_results_dir": "/home/aparcedo/IASEB/results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper"
}
# ---
# --- END CONTROL PANEL ---
# ---


# --- Category Definitions ---
st_coarse_categories = {1: "spatial", 2: "temporal", 3: "composite"}
entity_coarse_categories = {1: "human-human",
                            2: "human-object",
                            3: "human-animal",
                            4: "animal-animal",
                            5: "animal-object",
                            6: "object-object",
                            7: "human-self",
                            8: "no interaction"}

# %%
# --- 1. LOAD DATA BASED ON MODE ---

mode = CONTROL_PANEL["DATA_MODE"]
if mode == "st":
    data_path = CONTROL_PANEL["st_data_path"]
    coarse_categories = st_coarse_categories
    print("Running in 'spatiotemporal' (st) mode.")
elif mode == "entity":
    data_path = CONTROL_PANEL["entity_data_path"]
    coarse_categories = entity_coarse_categories
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

# READ DALTON RESULTS
print("Processing Dalton results...")
BASE_DIR = CONTROL_PANEL["dalton_results_dir"]
filenames = os.listdir(BASE_DIR)
ff_filenames = [fn for fn in filenames if "freeform" in fn]

for ff_fn in ff_filenames:
    data = json.load(open(os.path.join(BASE_DIR, ff_fn), 'r'))
    for sample_dict in tqdm(data, desc=f"Dalton: {ff_fn}", leave=False):
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        category_str = cls_data.get(caption)

        if not category_str:
            continue
        
        processed_records.append({
            "model": ff_fn.split("_")[0],
            "dataset": ff_fn.split("_")[1],
            "caption": caption,
            "coarse_category": coarse_categories[int(category_str[0])],
            "fine_category": category_str,
            "mvIoU": sample_dict["mvIoU"]
        })

# READ ALEJANDRO RESULTS
print("Processing Alejandro results...")
BASE_DIR = CONTROL_PANEL["alejandro_results_dir"]
filenames = os.listdir(BASE_DIR)
ff_filenames = [fn for fn in filenames if "freeform" in fn]

cnt = 0
for ff_fn in ff_filenames:
    data = json.load(open(os.path.join(BASE_DIR, ff_fn), 'r'))
    for sample_dict in tqdm(data["results"], desc=f"Alejandro: {ff_fn}", leave=False):
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        category_str = cls_data.get(caption)
        
        if not category_str:
            continue
        else:
            cnt += 1

        processed_records.append({
            "model": ff_fn.split("_")[1],
            "dataset": ff_fn.split("_")[2],
            "caption": caption,
            "coarse_category": coarse_categories[int(category_str[0])],
            "fine_category": category_str,
            "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))
        })

df = pd.DataFrame(processed_records)
print(f'Length of data frame: {len(df)}')
print(f'Length of captions categorized: {cnt}')

# %%
# --- 3. RADAR CHART FUNCTION (Unchanged) ---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_and_save_radar_chart(df, save_path=None, category_col='fine_category', color_by_model=True):
    """
    Generates and optionally saves a radar chart from a DataFrame.
    """

    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in the DataFrame.")
        
    plot_df = df.dropna(subset=[category_col])
    
    if plot_df.empty:
        print(f"Skipping chart for '{category_col}': No data found.")
        return

    categories = plot_df[category_col].unique()
    categories.sort()
    models = plot_df['model'].unique()
    models.sort()
    
    if len(categories) < 3:
        print(f"Skipping chart for '{category_col}': Not enough categories (need at least 3).")
        return

    pivot_df = plot_df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0)

    max_iou = pivot_df.values.max()
    radar_max = max_iou + 0.05
    if radar_max == 0.05: radar_max = 1.0

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    figsize = (10, 10) if num_vars < 15 else (min(30, num_vars * 0.8), min(30, num_vars * 0.8))
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist()
        values += values[:1]

        if color_by_model:
            color = plt.cm.get_cmap('tab10')(i % 10)
        else:
            color = 'blue'

        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    
    # xtick_fontsize = 22 if num_vars < 15 else 20
    xtick_fontsize = 15
    ax.set_xticklabels(categories, fontsize=xtick_fontsize)

    ax.set_yticks(np.arange(0, radar_max, radar_max / 5))
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, radar_max, radar_max / 5)], color="grey", size=15)
    ax.set_ylim(0, radar_max)
    
    ax.set_title(f'Mean mvIoU by {category_col} and Model', va='bottom', fontsize=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.08, 0.835), fontsize=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Radar chart saved to {save_path}")
    else:
        plt.show()

# %%
# --- 4. CREATE AND SAVE PLOTS ---

print(f"Creating charts for '{mode}' mode...")

# 1. Plot by COARSE category
create_and_save_radar_chart(
    df, 
    save_path=CONTROL_PANEL['coarse_chart_output'], 
    category_col='coarse_category', 
    color_by_model=True
)

# 2. Plot by FINE category
# create_and_save_radar_chart(
#     df, 
#     save_path=CONTROL_PANEL['fine_chart_output'], 
#     category_col='fine_category', 
#     color_by_model=True
# )

print("All charts created.")
# %%