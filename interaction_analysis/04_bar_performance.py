# %%
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import json
import pandas as pd
import os
import io
from tqdm import tqdm 
import seaborn as sns
# %%
st_data_path = "/home/aparcedo/IASEB/clustering/hcstvg12_vidvrdstg_gpt4omini_st_class_3200sample_v1.csv"
st_classification_data = pd.read_csv(st_data_path)
st_cls_data = dict(zip(
            [
            caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                for caption in st_classification_data["caption"]],
            st_classification_data["category"]
            ))
# %%
BASE_DIR = "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper"

filenames = os.listdir(BASE_DIR)
ff_filenames = []
for fn in filenames:
    if "freeform" in fn:
        ff_filenames.append(fn)

# %%
coarse_categories = {1: "spatial", 2: "temporal", 3: "composite"}
processed_records = [] 
cnt = 0
# We need to load a lot of data
for ff_fn in ff_filenames:
    data = json.load(open(os.path.join(BASE_DIR, ff_fn), 'r'))
    for sample_dict in tqdm(data["results"]):
        caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        st_cat = st_cls_data.get(caption)
        if not st_cat:
            # print(f'caption not categorized')
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
            "mvIoU": sample_dict["mvIoU_tube_step"]
        })

df = pd.DataFrame(processed_records)
print(f'Length of data frame: {len(df)}')
print(f'Length of captions categorized: {cnt}')

# %%
def plot_coarse_performance(df, savepath=None):
    sns.set_theme(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="coarse_category",
        y="mvIoU",
        hue="model",
        palette="viridis", # A nice color palette
        order=list(coarse_categories.values())
    )

    # Add labels and title for clarity
    ax.set_title('Model Performance (mvIoU) by Coarse Category', fontsize=16)
    ax.set_xlabel('Coarse Category', fontsize=12)
    ax.set_ylabel('Mean mvIoU Score', fontsize=12)
    ax.set_ylim(0, 1.0) # Set y-axis limit for scores between 0 and 1

    # Show the plot
    plt.legend(title='Model')
    plt.tight_layout()
    if save_path: fig.savefig(save_path, format='svg', bbox_inches='tight')
plot_coarse_performance(df)

# %%

def plot_fine_performance(df, savepath=None):
    fine_category_order = sorted(df['fine_category'].unique())
    plt.figure(figsize=(20, 8)) 
    ax_fine = sns.barplot(
        data=df,
        x="fine_category",
        y="mvIoU",
        hue="model",
        palette="plasma",  # Using a different palette for variety
        order=fine_category_order # Use the sorted order
    )
    ax_fine.set_title('Model Performance (mvIoU) by Fine Category', fontsize=18)
    ax_fine.set_xlabel('Fine Category', fontsize=14)
    ax_fine.set_ylabel('Mean mvIoU Score', fontsize=14)
    ax_fine.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right', fontsize=10) 
    plt.legend(title='Model')
    plt.tight_layout() 
    if save_path: fig.savefig(save_path, format='svg', bbox_inches='tight')
plot_fine_performance(df)


# %%

# RADAR CHART
import numpy as np

# Calculate the mean mvIoU for each model and fine_category
avg_performance = df.groupby(['model', 'fine_category'])['mvIoU'].mean().unstack(fill_value=0)

# Get the list of fine categories (axes for the radar chart)
fine_categories = avg_performance.columns.tolist()
num_vars = len(fine_categories)

# Calculate angle for each category
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Complete the loop for plotting

# Create figure and polar plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Optional: Set common outer limit for the chart, slightly above max mvIoU
max_mvlou = avg_performance.values.max()
ax.set_ylim(0, max_mvlou * 1.1) # 10% buffer above max

# Plot each model's performance
for i, model in tqdm(enumerate(avg_performance.index)):
    values = avg_performance.loc[model].tolist()
    values += values[:1] # Complete the loop for plotting

    # Plot the line for the model
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model,
            color=plt.cm.viridis(i / len(avg_performance.index))) # Use a colormap

    # Fill the area beneath the line
    ax.fill(angles, values, alpha=0.1,
            color=plt.cm.viridis(i / len(avg_performance.index)))

    # Add labels to the points
    for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        ax.text(angle, value + 0.03, f'{value:.2f}',
                horizontalalignment='center', verticalalignment='center',
                size=10, color=plt.cm.viridis(i / len(avg_performance.index)))

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(fine_categories, fontsize=12)

# Add a title and legend
ax.set_title('Model Performance (mvIoU) by Fine Category (Radar Chart)', size=18, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Customize grid and ticks (optional, to match your example)
ax.set_yticklabels([]) # Hide radial ticks labels if you prefer to label points directly
ax.grid(True) # Show grid lines

plt.tight_layout()
plt.savefig('fine_category_radar_chart.png')
plt.show()