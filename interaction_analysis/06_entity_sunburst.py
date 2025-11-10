# %%
# plot_entity_sunburst.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.colors as mcolors

# --- Import constants ---
from constants import (
    ENTITY_COARSE_CATEGORIES_MAP,
    ENTITY_BASE_COLORMAPS,
    ENTITY_COARSE_COLOR_MAP,
    DEFAULT_FALLBACK_CMAP, 
    OUTER_RING_GRADIENT_RANGE,
    UNIVERSAL_FONTSIZE
)
# ---

# ---
# ---  ENTITY (DATA-DRIVEN) CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- File Paths ---
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv",
    "output_savename": "sunburst_entity_distribution.svg", 
    
    # --- Sunburst Chart Styling ---
    "hole_radius": 0.1,
    "inner_ring_radius": 0.7,  # Level 1 ring
    "outer_ring_radius": 1.6   # Level 2 ring
}
# ---
# ---  ENTITY (EQUAL-DIST) CONTROL PANEL  ---
# ---
EQUAL_DIST_CONTROL_PANEL = {
    "output_savename": "sunburst_entity_equal_dist.svg", 
    
    # --- Sunburst Chart Styling ---
    "hole_radius": 0.0,
    "inner_ring_radius": 0.8,
    "outer_ring_radius": 1.8,
}
# ---

# This is the 2-level hierarchy for Entities
ENTITY_HIERARCHY_TEXT = """
1.0 Human-Human
    1.1 Cooperative (e.g., helping, exchanging)
    1.2 Competitive (e.g., fighting, sports)
    1.3 Affective (e.g., hugging, arguing, kissing)
    1.4 Proximity (e.g., person A is behind person B, standing near, away)
    1.5 Observation (e.g., watching, looking at)
    1.6 Spatial (e.g., human/object A is larger/smaller than human/object B)
2.0 Human-Object
    2.1 Active Manipulation (e.g., opening, cutting, riding bicycle, holding, carrying)
    2.2 Proximity (e.g., person A is behind object B, standing near, away, pass)
    2.3 Passive (e.g., sitting, wearing)
    2.4 Spatial (e.g., human/object A is larger/smaller than human/object B)
3.0 Human-Animal
    3.1 Direct Interaction (e.g., petting, feeding, touching)
    3.2 Observation (e.g., watching)
    3.3 Proximity (e.g., standing near)
    3.4 Spatial (e.g., human/animal A is larger/smaller than human/animal B)
4.0 Animal-Animal
    4.1 Proximity (e.g., playing, standing near, to the right/left of)
    4.2 Antagonistic (e.g., fighting, hunting)
    4.3 Observation (e.g., watching, looking at)
    4.4 Spatial (e.g., animal A is larger/smaller than animal B)
5.0 Animal-Object
    5.1 Interaction (e.g., playing with toy, building nest)
    5.2 Proximity (e.g., standing near, away)
    5.3 Spatial (e.g., animal/object A is larger/smaller than animal/object B)
6.0 Object-Object
    6.1 Spatial/Movement (e.g., car moving near another car)
    6.2 Proximity (e.g., object A is beneath/above object B, object C is away/close from/to object D)
    6.3 Spatial (e.g., object A is larger/smaller than object B)
7.0 Human-Self
    7.1 Self-interaction (e.g., cover mouth, raise hand)
8.0 No Interaction
    8.1 A single agent or object acting in isolation.
"""

def parse_entity_hierarchy(hierarchy_text):
    """Parses the 2-level entity hierarchy text into mapping dicts."""
    l1_map = {}
    l2_map = {}
    
    current_l1_id = None
    
    for line in hierarchy_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Match Level 1 (e.g., "1.0 Human-Human")
        l1_match = re.match(r'^(\d+)\.0\s+(.*)', line)
        if l1_match:
            l1_id = l1_match.group(1)
            l1_name = f"{l1_id}.0 {l1_match.group(2)}"
            l1_map[l1_id] = l1_name
            l2_map[f"{l1_id}.0"] = l1_name # Add L1 to L2 map as its own entry
            current_l1_id = l1_id
            continue
            
        # Match Level 2 (e.g., "1.1 Cooperative...")
        l2_match = re.match(r'^(\d+\.\d+)\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l2_match and current_l1_id:
            l2_id = l2_match.group(1)
            l2_name = f"{l2_id} {l2_match.group(2)}"
            
            if l2_id.startswith(current_l1_id + '.'):
                l2_map[l2_id] = l2_name
            else:
                print(f"Warning: Skipping mismatch line: {line}")

    for l1_id, l1_name in l1_map.items():
        if f"{l1_id}.0" not in l2_map:
             l2_map[f"{l1_id}.0"] = l1_name
        if l1_id == '8':
            l2_map["8.1"] = "8.1 A single agent or object acting in isolation."

    return l1_map, l2_map


# =============================================================================
# --- DATA DISTRIBUTION PLOTTING (FROM CSV)
# =============================================================================

def create_sunburst_chart(savename, df, 
                          base_cmaps,
                          coarse_color_map,
                          default_cmap_obj,
                          hole_radius=0.1, 
                          inner_ring_radius=0.4, 
                          outer_ring_radius=1, 
                          outer_ring_gradient_range=[0.2, 0.6],
                          universal_fontsize=15):
    """
    Creates a 2-level, static sunburst chart using Matplotlib based on data counts.
    """
    
    # 1. --- Data Preparation ---
    df_inner = df.groupby('level1_cat').size().reset_index(name='counts')
    df_inner = df_inner.sort_values('level1_cat').reset_index(drop=True)
    df_inner['label'] = df_inner['level1_cat'].apply(lambda x: x[:20])
    
    df_outer = df.groupby(['level1_cat', 'level2_cat']).size().reset_index(name='counts')
    df_outer = df_outer.sort_values(['level1_cat', 'level2_cat']).reset_index(drop=True)
    df_outer['label'] = df_outer['level2_cat'].apply(lambda x: x[:20])

    # --- Create Inner Ring (L1) Colors ---
    inner_colors = df_inner['level1_cat'].map(coarse_color_map)
    if inner_colors.isna().any():
        missing = df_inner[inner_colors.isna()]['level1_cat'].unique()
        raise AssertionError(f"Missing L1 color keys: {missing}. Check keys in `coarse_color_map`.")

    # --- Create Outer Ring (L2) Colors (Hierarchically) ---
    fine_color_map = {}
    unique_coarse_categories = df_inner['level1_cat'].unique()
    
    for cat in unique_coarse_categories:
        cmap = base_cmaps.get(cat, default_cmap_obj)
        fine_cats_for_coarse = df_outer[df_outer['level1_cat'] == cat]['level2_cat'].unique()
        num_fine_cats = len(fine_cats_for_coarse)
        if num_fine_cats == 0: continue
            
        color_palette = [cmap(i) for i in np.linspace(outer_ring_gradient_range[0], 
                                                      outer_ring_gradient_range[1], 
                                                      num_fine_cats)]
        
        for i, fine_cat in enumerate(fine_cats_for_coarse):
            fine_color_map[fine_cat] = color_palette[i]
    
    outer_colors = df_outer['level2_cat'].map(fine_color_map)
    if outer_colors.isna().any():
        outer_colors = outer_colors.fillna('#808080')

    # --- MODIFICATION: Make L1-leaf slices transparent in L2 ring ---
    is_l1_leaf = (df_outer['level1_cat'] == df_outer['level2_cat'])
    num_leaves_to_set = is_l1_leaf.sum()
    if num_leaves_to_set > 0:
        transparent_list = [(0, 0, 0, 0)] * num_leaves_to_set
        outer_colors.loc[is_l1_leaf] = transparent_list
    
    outer_edgecolors = ['w'] * len(df_outer)
    for i, is_leaf in enumerate(is_l1_leaf): 
        if is_leaf:
            outer_edgecolors[i] = (0, 0, 0, 0)

    # 4. --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(aspect="equal"))
    plt.style.use('default') 
    base_wedge_props = dict(edgecolor='w', linewidth=1.5)
    text_props = dict(fontweight='normal', ha='center', va='center')
    MIN_FONT_SIZE = 8 
    REFERENCE_ANGLE = 8 
    
    outer_wedges, _ = ax.pie(
        df_outer['counts'], radius=outer_ring_radius, colors=outer_colors.tolist(),
        wedgeprops=dict(width=outer_ring_radius - inner_ring_radius, linewidth=1.5), 
        startangle=90,
    )
    for i, wedge in enumerate(outer_wedges):
        wedge.set_edgecolor(outer_edgecolors[i])

    inner_wedges, _ = ax.pie(
        df_inner['counts'], radius=inner_ring_radius, colors=inner_colors.tolist(),
        wedgeprops=dict(width=inner_ring_radius - hole_radius, **base_wedge_props), 
        startangle=90,
    )
    for i, p in enumerate(inner_wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1 
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = hole_radius + (inner_ring_radius - hole_radius) / 2
        ax.text(x * text_radius, y * text_radius, df_inner['label'][i],
                rotation=rotation, color='black', fontsize=universal_fontsize, 
                **text_props)
    for i, p in enumerate(outer_wedges):
        slice_angle = p.theta2 - p.theta1
        if slice_angle == 0: continue
        
        # --- MODIFICATION: Skip labeling transparent slices ---
        if df_outer['level1_cat'][i] == df_outer['level2_cat'][i]:
            continue
            
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = inner_ring_radius + (outer_ring_radius - inner_ring_radius) / 2
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
        ax.text(x * text_radius, y * text_radius, df_outer['label'][i],
                rotation=rotation, color='black', fontsize=dynamic_fontsize,  
                **text_props)               
    
    print(f"Saving static chart to {savename}...")
    fig.savefig(savename, format='svg', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    print("Done.")


def main_data_dist():
    """
    Main execution function to load data and plot the DATA distribution.
    """
    
    # --- 1. Setup Environment ---
    print("--- Running in 'entity' (Data Distribution) mode. ---")
    data_path = CONTROL_PANEL["entity_data_path"]
    
    L1_MAP, L2_MAP = parse_entity_hierarchy(ENTITY_HIERARCHY_TEXT)

    # --- FIX: Re-key the color maps to use display names ---
    plot_coarse_color_map = {}
    plot_base_cmaps = {}
    for l1_id_str, display_name in L1_MAP.items():
        base_name = ENTITY_COARSE_CATEGORIES_MAP[int(l1_id_str)]
        if base_name in ENTITY_COARSE_COLOR_MAP:
            plot_coarse_color_map[display_name] = ENTITY_COARSE_COLOR_MAP[base_name]
        if base_name in ENTITY_BASE_COLORMAPS:
            plot_base_cmaps[display_name] = ENTITY_BASE_COLORMAPS[base_name]

    # --- 2. Load Data ---
    print(f"Loading classification data from {data_path}")
    try:
        classification_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    print(f"Loaded {len(classification_df)} total annotations.")

    # --- 3. Process Data into DataFrame ---
    processed_records = [] 
    print("Processing annotations...")
    for _, row in classification_df.iterrows():
        entity_cat_str = row.get('category')
        if not entity_cat_str or pd.isna(entity_cat_str):
            continue
        full_id_str = str(entity_cat_str).split(" ")[0]
        id_parts = full_id_str.split('.')
        if len(id_parts) < 2:
            continue
        l1_id = id_parts[0]
        l2_id = full_id_str
        l1_name = L1_MAP.get(l1_id)
        l2_name = L2_MAP.get(l2_id)
        if not l1_name or not l2_name:
            continue
            
        # --- MODIFICATION: Treat 7.x and 8.x as L1 leaves ---
        if l1_id == '7' or l1_id == '8':
            l2_name = l1_name
        # --- END MODIFICATION ---
            
        processed_records.append({
            "level1_cat": l1_name,
            "level2_cat": l2_name,
        })
    df = pd.DataFrame(processed_records)
    if df.empty:
        print("No valid data processed. Exiting.")
        return
    print(f'Processed {len(df)} valid hierarchical annotations.')

    # --- 4. Create and Save Chart ---
    create_sunburst_chart(
        savename=CONTROL_PANEL['output_savename'], 
        df=df, 
        base_cmaps=plot_base_cmaps,
        coarse_color_map=plot_coarse_color_map,
        default_cmap_obj=DEFAULT_FALLBACK_CMAP,
        hole_radius=CONTROL_PANEL['hole_radius'], 
        inner_ring_radius=CONTROL_PANEL['inner_ring_radius'], 
        outer_ring_radius=CONTROL_PANEL['outer_ring_radius'],
        outer_ring_gradient_range=OUTER_RING_GRADIENT_RANGE,
        universal_fontsize=UNIVERSAL_FONTSIZE
    )

# =============================================================================
# --- EQUAL DISTRIBUTION PLOTTING (HIERARCHY-ONLY)
# =============================================================================

def generate_equal_dist_df(l1_map, l2_map):
    """
    Generates a 'finest_level' dataframe where counts are
    distributed equally at each level of the hierarchy.
    """
    processed_records = []
    
    l1_names = l1_map.values()
    if not l1_names:
        return pd.DataFrame()
        
    l1_base_count = 1.0 / len(l1_names)

    for l1_name in l1_names:
        l1_id_prefix = l1_name.split(" ")[0].split(".")[0] # e.g., "7"
        
        # --- MODIFICATION: Treat 7 and 8 as L1 leaves ---
        if l1_id_prefix == '7' or l1_id_prefix == '8':
            processed_records.append({
                "level1_cat": l1_name,
                "level2_cat": l1_name, # Set L2 to be L1
                "counts": l1_base_count
            })
            continue
        # --- END MODIFICATION ---
        
        # Find all L2 children for this L1, *excluding* the L1-self-entry
        l2_children = [name for id, name in l2_map.items() 
                       if id.startswith(l1_id_prefix + ".") and id != l1_id_prefix + ".0"]
        
        if not l2_children:
             print(f"Warning: No L2 children found for {l1_name}, treating as leaf.")
             processed_records.append({
                "level1_cat": l1_name,
                "level2_cat": l1_name,
                "counts": l1_base_count
             })
             continue
            
        l2_base_count = l1_base_count / len(l2_children)
        
        for l2_name in l2_children:
            processed_records.append({
                "level1_cat": l1_name,
                "level2_cat": l2_name,
                "counts": l2_base_count
            })

    return pd.DataFrame(processed_records)


def create_sunburst_chart_equal_dist(savename, df_finest, 
                                     base_cmaps,
                                     coarse_color_map,
                                     default_cmap_obj,
                                     hole_radius=0.1, 
                                     inner_ring_radius=0.4, 
                                     outer_ring_radius=1, 
                                     outer_ring_gradient_range=[0.2, 0.6],
                                     universal_fontsize=15):
    """
    Creates a 2-level, static sunburst chart using Matplotlib
    from a dataframe with pre-computed 'counts'.
    """
    
    # 1. --- Data Preparation ---
    df_outer = df_finest.sort_values(['level1_cat', 'level2_cat']).reset_index(drop=True)
    df_outer['label'] = df_outer['level2_cat'].apply(lambda x: x[:20])
    
    df_inner = df_outer.groupby('level1_cat')['counts'].sum().reset_index()
    df_inner = df_inner.sort_values('level1_cat').reset_index(drop=True)
    df_inner['label'] = df_inner['level1_cat'].apply(lambda x: x[:20])

    # --- Create Inner Ring (L1) Colors ---
    inner_colors = df_inner['level1_cat'].map(coarse_color_map)
    if inner_colors.isna().any():
        missing = df_inner[inner_colors.isna()]['level1_cat'].unique()
        raise AssertionError(f"Missing L1 color keys: {missing}. Check keys in `coarse_color_map`.")

    # --- Create Outer Ring (L2) Colors (Hierarchically) ---
    fine_color_map = {}
    unique_coarse_categories = df_inner['level1_cat'].unique()
    
    for cat in unique_coarse_categories:
        cmap = base_cmaps.get(cat, default_cmap_obj)
        fine_cats_for_coarse = df_outer[df_outer['level1_cat'] == cat]['level2_cat'].unique()
        num_fine_cats = len(fine_cats_for_coarse)
        if num_fine_cats == 0: continue
            
        color_palette = [cmap(i) for i in np.linspace(outer_ring_gradient_range[0], 
                                                      outer_ring_gradient_range[1], 
                                                      num_fine_cats)]
        
        for i, fine_cat in enumerate(fine_cats_for_coarse):
            fine_color_map[fine_cat] = color_palette[i]
    
    outer_colors = df_outer['level2_cat'].map(fine_color_map)
    if outer_colors.isna().any():
        outer_colors = outer_colors.fillna('#808080')

    # --- MODIFICATION: Make L1-leaf slices transparent in L2 ring ---
    is_l1_leaf = (df_outer['level1_cat'] == df_outer['level2_cat'])
    num_leaves_to_set = is_l1_leaf.sum()
    if num_leaves_to_set > 0:
        transparent_list = [(0, 0, 0, 0)] * num_leaves_to_set
        outer_colors.loc[is_l1_leaf] = transparent_list
    
    outer_edgecolors = ['w'] * len(df_outer)
    for i, is_leaf in enumerate(is_l1_leaf): 
        if is_leaf:
            outer_edgecolors[i] = (0, 0, 0, 0)

    # 4. --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(aspect="equal"))
    plt.style.use('default') 
    base_wedge_props = dict(edgecolor='w', linewidth=1.5)
    text_props = dict(fontweight='normal', ha='center', va='center')
    MIN_FONT_SIZE = 8 
    REFERENCE_ANGLE = 8 
    
    outer_wedges, _ = ax.pie(
        df_outer['counts'], radius=outer_ring_radius, colors=outer_colors.tolist(),
        wedgeprops=dict(width=outer_ring_radius - inner_ring_radius, linewidth=1.5), 
        startangle=90,
    )
    for i, wedge in enumerate(outer_wedges):
        wedge.set_edgecolor(outer_edgecolors[i])
        
    inner_wedges, _ = ax.pie(
        df_inner['counts'], radius=inner_ring_radius, colors=inner_colors.tolist(),
        wedgeprops=dict(width=inner_ring_radius - hole_radius, **base_wedge_props), 
        startangle=90,
    )
    for i, p in enumerate(inner_wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1 
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = hole_radius + (inner_ring_radius - hole_radius) / 2
        ax.text(x * text_radius, y * text_radius, df_inner['label'][i],
                rotation=rotation, color='black', fontsize=universal_fontsize, 
                **text_props)
    for i, p in enumerate(outer_wedges):
        slice_angle = p.theta2 - p.theta1
        if slice_angle == 0: continue
        
        # --- MODIFICATION: Skip labeling transparent slices ---
        if df_outer['level1_cat'][i] == df_outer['level2_cat'][i]:
            continue
            
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = inner_ring_radius + (outer_ring_radius - inner_ring_radius) / 2
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
        ax.text(x * text_radius, y * text_radius, df_outer['label'][i],
                rotation=rotation, color='black', fontsize=dynamic_fontsize,  
                **text_props)               
    
    print(f"Saving static chart to {savename}...")
    fig.savefig(savename, format='svg', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    print("Done.")


def main_equal_dist():
    """
    Main execution function to generate and plot the EQUAL distribution chart.
    """
    
    # --- 1. Setup Environment ---
    print("--- Running in 'entity' (Equal Distribution) mode. ---")
    
    L1_MAP, L2_MAP = parse_entity_hierarchy(ENTITY_HIERARCHY_TEXT)
    print(f"Found {len(L1_MAP)} L1 and {len(L2_MAP)} L2 categories.")
    
    # Re-key the color maps
    plot_coarse_color_map = {}
    plot_base_cmaps = {}
    for l1_id_str, display_name in L1_MAP.items():
        base_name = ENTITY_COARSE_CATEGORIES_MAP[int(l1_id_str)]
        if base_name in ENTITY_COARSE_COLOR_MAP:
            plot_coarse_color_map[display_name] = ENTITY_COARSE_COLOR_MAP[base_name]
        if base_name in ENTITY_BASE_COLORMAPS:
            plot_base_cmaps[display_name] = ENTITY_BASE_COLORMAPS[base_name]

    # --- 2. Generate Data ---
    print("Generating equal distribution data...")
    df_finest = generate_equal_dist_df(L1_MAP, L2_MAP)
    
    if df_finest.empty:
        print("No data generated. Exiting.")
        return
        
    print(f'Generated {len(df_finest)} finest-level records with equal counts.')

    # --- 3. Create and Save Chart ---
    create_sunburst_chart_equal_dist(
        savename=EQUAL_DIST_CONTROL_PANEL['output_savename'], 
        df_finest=df_finest, 
        base_cmaps=plot_base_cmaps,
        coarse_color_map=plot_coarse_color_map,
        default_cmap_obj=DEFAULT_FALLBACK_CMAP,
        hole_radius=EQUAL_DIST_CONTROL_PANEL['hole_radius'],
        inner_ring_radius=EQUAL_DIST_CONTROL_PANEL['inner_ring_radius'], 
        outer_ring_radius=EQUAL_DIST_CONTROL_PANEL['outer_ring_radius'],
        outer_ring_gradient_range=OUTER_RING_GRADIENT_RANGE,
        universal_fontsize=UNIVERSAL_FONTSIZE
    )

# =============================================================================
# --- SCRIPT ENTRYPOINT ---
# =============================================================================

if __name__ == "__main__":
    # 1. Run the plot for the real data distribution
    main_data_dist()
    
    print("\n" + "="*50 + "\n")
    
    # 2. Run the plot for the equal (hierarchy) distribution
    main_equal_dist()
# %%