# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm  # type: ignore
# %%
st_data_path = "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv"
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


def create_sunburst_chart(df, hole_radius=0.1, inner_ring_radius=0.4, outer_ring_radius=1, output_filename="static_sunburst_chart.png"):
    """
    Creates a hierarchical, static sunburst chart using Matplotlib
    and professional, built-in sequential color palettes.

    - Static Image: Saves as a high-resolution PNG.
    - Hierarchy:
        - Center: 20% white hole (small).
        - Inner Ring: 'coarse_category' (thinner, 3 base colors).
        - Outer Ring: 'fine_category' (thicker, shades of parent).
    - Styling:
        - Text is radial, bold, and shows the first word only.
        - No percentages are shown.
        - Slices have white outlines.

    Args:
        df (pd.DataFrame): The input dataframe. Must contain
                           'coarse_category' and 'fine_category' columns.
        output_filename (str): The name of the file to save (e.g., "chart.png").
    """
    
    # 1. --- Data Preparation ---
    
    # Data for the Inner Ring (coarse_category)
    df_inner = df.groupby('coarse_category').size().reset_index(name='counts')
    df_inner['label'] = df_inner['coarse_category'].apply(lambda x: x.split()[0])
    
    # Data for the Outer Ring (fine_category)
    # IMPORTANT: We must sort by 'coarse_category' so colors align
    df_outer = df.groupby(['coarse_category', 'fine_category']).size().reset_index(name='counts')
    df_outer['label'] = df_outer['fine_category']

    
    # 2. --- (NEW) Hierarchical Color Mapping ---
    
    # Define 3 base *colormaps* (palettes) for the inner ring
    # These are standard Matplotlib palettes.
    base_cmap_names = {
        'spatial': 'Purples',
        'composite': 'Oranges',
        'temporal': 'Greens',
    }
    default_cmap = 'Greys'

    # --- Create Inner Ring Colors ---
    coarse_color_map = {}
    coarse_categories = df_inner['coarse_category'].unique()
    
    for cat in coarse_categories:
        cmap_name = base_cmap_names.get(cat, default_cmap)
        cmap = plt.get_cmap(cmap_name)
        # Pick a strong, dark color from the map for the inner ring
        coarse_color_map[cat] = cmap(0.7) 
        
    inner_colors = df_inner['coarse_category'].map(coarse_color_map)

    # --- Create Outer Ring Colors (Hierarchically) ---
    fine_color_map = {}
    for cat in coarse_categories:
        # Get the same colormap as the parent
        cmap_name = base_cmap_names.get(cat, default_cmap)
        cmap = plt.get_cmap(cmap_name)
        
        # Find all fine_categories for *this* coarse_category
        fine_cats_for_coarse = df[df['coarse_category'] == cat]['fine_category'].unique()
        num_fine_cats = len(fine_cats_for_coarse)
        
        # Generate a list of *light* shades from that colormap
        # We use np.linspace to pick N colors evenly from the "light" end (0.2 to 0.6)
        color_palette = [cmap(i) for i in np.linspace(0.2, 0.6, num_fine_cats)]
        
        # Map each fine_category to one of these light shades
        for i, fine_cat in enumerate(fine_cats_for_coarse):
            fine_color_map[fine_cat] = color_palette[i]
    
    # Map the outer ring colors from the new hierarchical map
    outer_colors = df_outer['fine_category'].map(fine_color_map)
    outer_colors = outer_colors.fillna('#808080') # Fallback grey

    # 4. --- Plotting ---
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))
    plt.style.use('default') 

    wedge_props = dict(edgecolor='w', linewidth=1.5)
    text_props = dict(fontweight='normal', ha='center', va='center')
    BASE_FONT_SIZE = 14
    COARSE_FONTSIZE = 14
    MIN_FONT_SIZE = 8
    REFERENCE_ANGLE = 8
    

    # --- Draw Outer Ring (fine_category, shades, thick) ---
    outer_wedges, _ = ax.pie(
        df_outer['counts'],
        radius=outer_ring_radius, # Outer radius = 1.0
        colors=outer_colors,
        wedgeprops=dict(width=outer_ring_radius - inner_ring_radius, **wedge_props), 
        startangle=90,
    )

    # --- Draw Inner Ring (coarse_category, 3 colors, thin) ---
    inner_wedges, _ = ax.pie(
        df_inner['counts'],
        radius=inner_ring_radius, # Outer radius = 0.5
        colors=inner_colors,
        wedgeprops=dict(width=inner_ring_radius - hole_radius, **wedge_props), 
        startangle=90,
    )

        
    # Add labels for the inner ring (coarse_category)
    for i, p in enumerate(inner_wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1 
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        rotation = ang
        if 90 < ang < 270: 
            rotation = ang + 180
            
        text_radius = hole_radius + (inner_ring_radius - hole_radius) / 2
        ax.text(x * text_radius,
                y * text_radius,
                df_inner['label'][i],
                rotation=rotation, 
                color='black',
                fontsize=COARSE_FONTSIZE,
                **text_props)


    # Add labels for the outer ring (fine_category)
    for i, p in enumerate(outer_wedges):
        slice_angle = p.theta2 - p.theta1
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        
        rotation = ang
        if 90 < ang < 270:
            rotation = ang + 180
        
        text_radius = inner_ring_radius + (outer_ring_radius - inner_ring_radius) / 2
        
        # --- Existing logic for text color ---
        label_color = 'black' 
        scaled_fontsize = BASE_FONT_SIZE * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, BASE_FONT_SIZE))
            
        ax.text(x * text_radius, y * text_radius,
                df_outer['label'][i],
                rotation=rotation, 
                color=label_color,
                fontsize=dynamic_fontsize,  
                **text_props)               
    
    # 6. --- Save Static Image ---
    print(f"Saving static chart to {output_filename}...")
    # fig.savefig(output_filename, dpi=600, bbox_inches='tight')
    fig.savefig("sunburst_hcstvg12_vidvrdstg_gpt4omini_st_class_v1.svg", format='svg', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    print("Done.")
    # plt.close(fig) # Close the figure to free memory

create_sunburst_chart(df, hole_radius=0.1, inner_ring_radius=0.6, outer_ring_radius=1.5)
# %%

