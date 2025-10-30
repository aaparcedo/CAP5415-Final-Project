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
# Read spatiotemporal classification data
st_data_path = "/home/aparcedo/IASEB/clustering/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv"
st_classification_data = pd.read_csv(st_data_path)
st_cls_data = dict(zip(
            [
            caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                for caption in st_classification_data["caption"]],
            st_classification_data["category"]
            ))

# %%
# Now, we're going to explore the results that will be in our paper
# Let's begin by visualization the distribution of our datasets

# Find all final freeform results
BASE_DIR = "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper"
filenames = os.listdir(BASE_DIR)
ff_filenames = []
for fn in filenames:
    if "freeform" in fn:
        ff_filenames.append(fn)

# %%
# Aggreagate results from JSONs into a single Pandas DataFrame
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
import plotly.express as px
import plotly.io as pio # Required for write_image

def plot_dataset_distribution(df, save_path=None):
    """
    Shows and optionally saves a publication-ready pie chart of dataset distribution.
    
    NOTE: Requires the 'kaleido' package to save static images.
    Install it with: pip install kaleido
    """
    
    # Use a donut chart (hole=0.4) as it's often considered cleaner
    fig = px.pie(
        df,
        names='dataset',
        # title='Dataset Distribution', # Title is removed; use a figure caption in your paper
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.00
    )

    # Update traces for a clean, professional look
    fig.update_traces(
        textposition='outside',  # Move labels outside the slices for better clarity
        textinfo='percent+label+value',
        textfont_size=16,        # Increase font size for readability
        marker=dict(line=dict(color='#000000', width=1.5)) # Add a crisp black border to slices
    )

    # Update layout for a minimal, publication-ready appearance
    fig.update_layout(
        showlegend=False,  # Hide the legend (info is already in 'textinfo')
        font=dict(
            family="Arial, sans-serif", # Use a common sans-serif font
            size=18,                   # Set base font size
            color="#000000"            # Set font color to black
        ),
        # Set transparent background for easy placement in your paper
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Adjust margins to ensure outside labels fit completely
        margin=dict(t=20, b=20, l=20, r=20) 
    )

    # --- Corrected Save Method ---
    if save_path:
        # Ensure the .svg extension is present for a vector graphic
        if not save_path.lower().endswith('.svg'):
            save_path = f"{save_path}.svg"
            
        # Use fig.write_image() for Plotly, not fig.savefig()
        try:
            # We can set a scale factor for higher resolution, though SVG is vector-based
            fig.write_image(save_path, scale=2) 
            print(f"Figure saved to {save_path}")
        except ValueError as e:
            print(f"Error saving figure: {e}")
            print("This usually means the 'kaleido' package is not installed.")
            print("Please run: pip install kaleido")
            
    return fig # Return the figure object (e.g., to display in a notebook)
    
plot_dataset_distribution(st_classification_data["dataset"], save_path='hcstvg12_vidvrdstg_gpt4omini_st_class_v1_pie_distribution')

# %%

