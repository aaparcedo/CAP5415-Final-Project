# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm  # type: ignore
import matplotlib.colors as mcolors

# --- Import constants ---
from constants import (
    ST_COARSE_CATEGORIES_MAP, ENTITY_COARSE_CATEGORIES_MAP,
    ST_BASE_COLORMAPS, ENTITY_BASE_COLORMAPS,
    ST_COARSE_COLOR_MAP, ENTITY_COARSE_COLOR_MAP,
    DEFAULT_FALLBACK_CMAP, OUTER_RING_GRADIENT_RANGE,
    UNIVERSAL_FONTSIZE # <-- IMPORT THE FONTSIZE
)
# ---

# ---
# ---  SINGLE CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- Mode Switch ---
    "DATA_MODE": "st",  # <-- CHANGE THIS: "st" or "entity"

    # --- File Paths ---
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrd_stg_gpt4omini_st_class_v1.csv",
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv",
    "base_dir": "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper",
    "output_savename": "sunburst_st_chart_v1.svg", 
    
    # --- Sunburst Chart Styling ---
    "hole_radius": 0.1,
    "inner_ring_radius": 0.6,
    "outer_ring_radius": 1.5
}
# ---
# --- END CONTROL PANEL ---
# ---


def create_sunburst_chart(savename, df, 
                          base_cmaps,
                          coarse_color_map,
                          default_cmap_obj,
                          hole_radius=0.1, 
                          inner_ring_radius=0.4, 
                          outer_ring_radius=1, 
                          outer_ring_gradient_range=[0.2, 0.6],
                          universal_fontsize=15): # <-- ADDED PARAMETER
    """
    Creates a hierarchical, static sunburst chart using Matplotlib.
    """
    
    # (Data prep is unchanged)
    df_inner = df.groupby('coarse_category').size().reset_index(name='counts')
    df_inner['label'] = df_inner['coarse_category']
    df_outer = df.groupby(['coarse_category', 'fine_category']).size().reset_index(name='counts')
    df_outer['label'] = df_outer['fine_category']
    inner_colors = df_inner['coarse_category'].map(coarse_color_map)
    fine_color_map = {}
    unique_coarse_categories = df_inner['coarse_category'].unique()
    for cat in unique_coarse_categories:
        cmap = base_cmaps.get(cat, default_cmap_obj) 
        fine_cats_for_coarse = df[df['coarse_category'] == cat]['fine_category'].unique()
        num_fine_cats = len(fine_cats_for_coarse)
        color_palette = [cmap(i) for i in np.linspace(outer_ring_gradient_range[0], 
                                                      outer_ring_gradient_range[1], 
                                                      num_fine_cats)]
        for i, fine_cat in enumerate(fine_cats_for_coarse):
            fine_color_map[fine_cat] = color_palette[i]
    outer_colors = df_outer['fine_category'].map(fine_color_map)
    outer_colors = outer_colors.fillna('#808080')

    # 4. --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))
    plt.style.use('default') 

    wedge_props = dict(edgecolor='w', linewidth=1.5)
    text_props = dict(fontweight='normal', ha='center', va='center')
    
    # --- FONTSIZE MODIFICATION ---
    # BASE_FONT_SIZE = 14 # <-- Replaced
    # COARSE_FONTSIZE = 14 # <-- Replaced
    MIN_FONT_SIZE = 8 # Keep a minimum
    REFERENCE_ANGLE = 8 # Keep for scaling
    
    # --- Draw Outer Ring ---
    outer_wedges, _ = ax.pie(
        df_outer['counts'],
        radius=outer_ring_radius,
        colors=outer_colors,
        wedgeprops=dict(width=outer_ring_radius - inner_ring_radius, **wedge_props), 
        startangle=90,
    )

    # --- Draw Inner Ring ---
    inner_wedges, _ = ax.pie(
        df_inner['counts'],
        radius=inner_ring_radius,
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
                fontsize=universal_fontsize, # <-- USE FONTSIZE
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
        label_color = 'black' 
        
        # --- FONTSIZE MODIFICATION ---
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
            
        ax.text(x * text_radius, y * text_radius,
                df_outer['label'][i],
                rotation=rotation, 
                color=label_color,
                fontsize=dynamic_fontsize,  # <-- USE DYNAMIC FONTSIZE
                **text_props)               
    
    # 6. --- Save Static Image ---
    print(f"Saving static chart to {savename}...")
    fig.savefig(savename, format='svg', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    print("Done.")


def main():
    """
    Main execution function to load data, generate colors, and create the chart.
    """
    
    # --- 1. Load Data based on DATA_MODE ---
    mode = CONTROL_PANEL["DATA_MODE"]
    if mode == "st":
        data_path = CONTROL_PANEL["st_data_path"]
        coarse_categories_map = ST_COARSE_CATEGORIES_MAP
        base_cmaps = ST_BASE_COLORMAPS
        coarse_color_map = ST_COARSE_COLOR_MAP
        print("Running in 'spatiotemporal' (st) mode.")
    elif mode == "entity":
        data_path = CONTROL_PANEL["entity_data_path"]
        coarse_categories_map = ENTITY_COARSE_CATEGORIES_MAP
        base_cmaps = ENTITY_BASE_COLORMAPS
        coarse_color_map = ENTITY_COARSE_COLOR_MAP
        print("Running in 'entity' mode.")
    else:
        raise ValueError(f"DATA_MODE in CONTROL_PANEL must be 'st' or 'entity', not '{mode}'")

    # (Data loading is unchanged)
    print(f"Loading data from {data_path}")
    classification_data = pd.read_csv(data_path)
    classification_data = dict(zip(
                [
                caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                    for caption in classification_data["caption"]],
                classification_data["category"]
                ))
    print(f"Scanning for files in {CONTROL_PANEL['base_dir']}")
    filenames = os.listdir(CONTROL_PANEL['base_dir'])
    ff_filenames = [fn for fn in filenames if "freeform" in fn]

    # --- 3. Process Data into DataFrame ---
    processed_records = [] 
    cnt = 0
    print("Processing result files...")
    for ff_fn in ff_filenames:
        data = json.load(open(os.path.join(CONTROL_PANEL['base_dir'], ff_fn), 'r'))
        for sample_dict in tqdm(data["results"], desc=f"Processing {ff_fn}", leave=False):
            caption = sample_dict["caption"].strip().lower().replace('.', '').replace('"', '').replace('\\', '')
            st_cat = classification_data.get(caption)
            if not st_cat:
                continue
            else:
                cnt += 1
            
            processed_records.append({
                "model": ff_fn.split("_")[1],
                "dataset": ff_fn.split("_")[2],
                "caption": caption,
                "coarse_category": coarse_categories_map[int(st_cat[0])], 
                "fine_category": st_cat,
                "coarse_id": st_cat[0],
                "fine_id": st_cat.split(" ")[0],
                "mvIoU": sample_dict["mvIoU_tube_step"]
            })

    df = pd.DataFrame(processed_records)
    print(f'Length of data frame: {len(df)}')
    print(f'Length of captions categorized: {cnt}')

    # --- 4. Create and Save Chart ---
    create_sunburst_chart(
        savename=CONTROL_PANEL['output_savename'], 
        df=df, 
        base_cmaps=base_cmaps,
        coarse_color_map=coarse_color_map,
        default_cmap_obj=DEFAULT_FALLBACK_CMAP,
        hole_radius=CONTROL_PANEL['hole_radius'], 
        inner_ring_radius=CONTROL_PANEL['inner_ring_radius'], 
        outer_ring_radius=CONTROL_PANEL['outer_ring_radius'],
        outer_ring_gradient_range=OUTER_RING_GRADIENT_RANGE,
        universal_fontsize=UNIVERSAL_FONTSIZE # <-- PASS THE CONSTANT
    )

if __name__ == "__main__":
    main()
# %%