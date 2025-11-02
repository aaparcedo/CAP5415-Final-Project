# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from tqdm import tqdm  # type: ignore
import colorsys
import matplotlib.colors as mcolors

# ---
# ---  SINGLE CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- Mode Switch ---
    "DATA_MODE": "st",  # <-- CHANGE THIS: "st" or "entity"

    # --- File Paths ---
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv",
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv",
    "base_dir": "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper",
    "output_savename": "sunburst_st_chart_v1.svg", # <-- Changed for "st"

    # --- HSL Color Gradient Generation ---
    "start_angle_degrees": 30.0,
    "lightness_start": 0.6,
    "lightness_end": 0.85,
    "sat_start": 0.7,
    "sat_end": 0.4,
    
    # --- Sunburst Chart Styling ---
    "hole_radius": 0.1,
    "inner_ring_radius": 0.6,
    "outer_ring_radius": 1.5,
    
    # --- Sunburst Color Sampling ---
    "inner_ring_gradient_pos": 0.1,
    "outer_ring_gradient_range": [0.7, 1.0]
}
# ---
# --- END CONTROL PANEL ---
# ---


def create_hsl_colormaps(category_names, 
                         start_angle_degrees=0,
                         lightness_start=0.5,
                         lightness_end=0.5,
                         sat_start=0.2,
                         sat_end=0.6):
    """
    Generates a dict of distinct colormaps based on HSL properties.
    
    Each colormap has a unique base Hue (evenly spaced) and a
    gradient defined by the start/end lightness and saturation values.
    """
    num_categories = len(category_names)
    if num_categories == 0:
        return {}
        
    cmap_dict = {}
    start_hue = start_angle_degrees / 360.0 # Convert degrees (0-360) to HLS scale (0-1)

    for i, category_name in enumerate(category_names):
        # 1. Get the base hue for this category (generalized for num_categories)
        hue = (start_hue + (i / float(num_categories))) % 1.0        
        
        # 2. Define start/end of gradient in HLS and convert to RGB
        start_rgb = colorsys.hls_to_rgb(hue, lightness_start, sat_start)
        end_rgb = colorsys.hls_to_rgb(hue, lightness_end, sat_end)
        
        # 3. Create the new colormap
        cmap_name = f'hls_grad_{i}'
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [start_rgb, end_rgb])
        
        # 4. Add it to the dictionary
        cmap_dict[category_name] = cmap
        
    return cmap_dict


def create_sunburst_chart(savename, df, 
                          base_cmaps,
                          default_cmap_obj,
                          coarse_categories,
                          hole_radius=0.1, 
                          inner_ring_radius=0.4, 
                          outer_ring_radius=1, 
                          inner_ring_gradient_pos=0.7,
                          outer_ring_gradient_range=[0.2, 0.6]):
    """
    Creates a hierarchical, static sunburst chart using Matplotlib.
    """
    
    # 1. --- Data Preparation ---
    df_inner = df.groupby('coarse_category').size().reset_index(name='counts')
    df_inner['label'] = df_inner['coarse_category']
    
    df_outer = df.groupby(['coarse_category', 'fine_category']).size().reset_index(name='counts')
    df_outer['label'] = df_outer['fine_category']

    # --- Create Inner Ring Colors ---
    coarse_color_map = {}
    unique_coarse_categories = df_inner['coarse_category'].unique()
    
    for cat in unique_coarse_categories:
        cmap = base_cmaps.get(cat, default_cmap_obj) 
        coarse_color_map[cat] = cmap(inner_ring_gradient_pos)
        
    inner_colors = df_inner['coarse_category'].map(coarse_color_map)

    # --- Create Outer Ring Colors (Hierarchically) ---
    fine_color_map = {}
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
        radius=outer_ring_radius,
        colors=outer_colors,
        wedgeprops=dict(width=outer_ring_radius - inner_ring_radius, **wedge_props), 
        startangle=90,
    )

    # --- Draw Inner Ring (coarse_category, 3 colors, thin) ---
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
    print(f"Saving static chart to {savename}...")
    fig.savefig(savename, format='svg', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    print("Done.")


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

def main():
    """
    Main execution function to load data, generate colors, and create the chart.
    """
    
    # --- 1. Load Data based on DATA_MODE ---
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
    classification_data = dict(zip(
                [
                caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
                    for caption in classification_data["caption"]],
                classification_data["category"]
                ))
    
    print(f"Scanning for files in {CONTROL_PANEL['base_dir']}")
    filenames = os.listdir(CONTROL_PANEL['base_dir'])
    ff_filenames = []
    for fn in filenames:
        if "freeform" in fn:
            ff_filenames.append(fn)

    # --- 2. Setup Categories & Colors ---
    # This is now general and works for 'st' (3) or 'entity' (8)
    category_names = list(coarse_categories.values())
    print(f"Generating {len(category_names)} colormaps...")
    
    base_cmaps = create_hsl_colormaps(
        category_names,
        start_angle_degrees=CONTROL_PANEL['start_angle_degrees'],
        lightness_start=CONTROL_PANEL['lightness_start'],
        lightness_end=CONTROL_PANEL['lightness_end'],
        sat_start=CONTROL_PANEL['sat_start'],
        sat_end=CONTROL_PANEL['sat_end']
    )
    
    default_cmap_obj = plt.get_cmap('Greys')

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
            
            # This logic works for both 'st' and 'entity'
            processed_records.append({
                "model": ff_fn.split("_")[1],
                "dataset": ff_fn.split("_")[2],
                "caption": caption,
                "coarse_category": coarse_categories[int(st_cat[0])],
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
        default_cmap_obj=default_cmap_obj,
        coarse_categories=coarse_categories,
        hole_radius=CONTROL_PANEL['hole_radius'], 
        inner_ring_radius=CONTROL_PANEL['inner_ring_radius'], 
        outer_ring_radius=CONTROL_PANEL['outer_ring_radius'],
        inner_ring_gradient_pos=CONTROL_PANEL['inner_ring_gradient_pos'],
        outer_ring_gradient_range=CONTROL_PANEL['outer_ring_gradient_range']
    )

if __name__ == "__main__":
    main()
# %%
