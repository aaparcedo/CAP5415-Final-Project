# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm 

# %%
# Read spatiotemporal classification data
st_data_fp = "/home/aparcedo/IASEB/clustering/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv"
st_classification_data = pd.read_csv(st_data_fp)
st_cls_data = dict(zip(
            [
            caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                for caption in st_classification_data["caption"]],
            st_classification_data["category"]
            ))


# %%
coarse_categories = {1: "spatial", 2: "temporal", 3: "composite"}
processed_records = [] 
# %%
# READ DALTON RESULTS
BASE_DIR = "/home/aparcedo/IASEB/results/llava_gdino_dalton_interpolated_results"
filenames = os.listdir(BASE_DIR)
ff_filenames = []
for fn in filenames:
    if "freeform" in fn:
        ff_filenames.append(fn)
cnt = 0

for ff_fn in ff_filenames:
    data = json.load(open(os.path.join(BASE_DIR, ff_fn), 'r'))
    for sample_dict in tqdm(data):
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        st_cat = st_cls_data.get(caption)

        processed_records.append({
            "model": ff_fn.split("_")[0], # Assumes model is the first part
            "dataset": ff_fn.split("_")[1], # Assumes dataset is the second part
            "caption": caption,
            "coarse_category": coarse_categories[int(st_cat[0])],
            "fine_category": " ".join(st_cat.split(" ")[1:]),
            "coarse_id": st_cat[0],
            "fine_id": st_cat.split(" ")[0],
            "mvIoU": sample_dict["mvIoU"]
        })
# %%
# READ ALEJANDRO RESULTS
BASE_DIR = "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper"
filenames = os.listdir(BASE_DIR)
ff_filenames = []
for fn in filenames:
    if "freeform" in fn:
        ff_filenames.append(fn)
# %%
cnt = 0

for ff_fn in ff_filenames:
    data = json.load(open(os.path.join(BASE_DIR, ff_fn), 'r'))
    for sample_dict in tqdm(data["results"]):
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        st_cat = st_cls_data.get(caption)
        
        if not st_cat:
            print(f'caption not categorized')
            continue
        else:
            cnt += 1

        processed_records.append({
            "model": ff_fn.split("_")[1], # Assumes model is the first part
            "dataset": ff_fn.split("_")[2], # Assumes dataset is the second part
            "caption": caption,
            "coarse_category": coarse_categories[int(st_cat[0])],
            "fine_category": " ".join(st_cat.split(" ")[1:]),
            "coarse_id": st_cat[0],
            "fine_id": st_cat.split(" ")[0],
            "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step"))
        })

df = pd.DataFrame(processed_records)
print(f'Length of data frame: {len(df)}')
print(f'Length of captions categorized: {cnt}')

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_and_save_radar_chart(df, save_path=None, category_col='fine_category', color_by_model=True):
    """
    Generates and optionally saves a radar chart from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'model', 'mvIoU', and category columns.
        save_path (str, optional): The file path to save the chart. If None, the chart is displayed.
                                   Supports formats like 'svg', 'png', etc.
        category_col (str, optional): The column to use for categories on the radar chart axes
                                      ('fine_category' or 'coarse_category'). Defaults to 'fine_category'.
        color_by_model (bool, optional): If True, different models will have different colors.
                                         If False, all lines will be the same color. Defaults to True.
    """

    # Ensure the category column exists
    if category_col not in df.columns:
        raise ValueError(f"Category column '{category_col}' not found in the DataFrame.")

    # Get unique categories and models
    categories = df[category_col].unique()
    models = df['model'].unique()

    # Calculate mean mvIoU for each model across each category
    pivot_df = df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')

    # Handle missing categories for some models (fill with 0 or NaN depending on desired visualization)
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0)

    # Determine the max value for the radar chart radius
    max_iou = pivot_df.values.max()
    radar_max = max_iou + 0.05  # 5 points above the best model (assuming "points" means 0.05 for mvIoU)

    # Number of variables
    num_vars = len(categories)

    # Calculate angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each model
    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist()
        values += values[:1]  # Complete the circle

        # Assign colors
        if color_by_model:
            color = plt.cm.get_cmap('tab10')(i % 10)  # Use a colormap for distinct colors
        else:
            color = 'blue' # Default single color

        ax.plot(angles, values, color=color, linewidth=2, label=model)
        ax.fill(angles, values, color=color, alpha=0.25)

    # Set labels for each axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Set y-axis ticks and limits
    ax.set_yticks(np.arange(0, radar_max, radar_max / 5)) # Example: 5 ticks
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, radar_max, radar_max / 5)], color="grey", size=7)
    ax.set_ylim(0, radar_max)

    ax.set_title(f'Mean mvIoU by {category_col} and Model', va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        print(f"Radar chart saved to {save_path}")
    else:
        plt.show()

create_and_save_radar_chart(df, save_path='gdino_llava_cog_shikra_ferret_freeform_radar.svg', category_col='fine_category', color_by_model=True)

# %%
