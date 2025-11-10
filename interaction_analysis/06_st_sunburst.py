# %%
# plot_st_sunburst.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.colors as mcolors

# --- Import constants ---
from constants import (
    ST_COARSE_CATEGORIES_MAP,
    ST_BASE_COLORMAPS,
    ST_COARSE_COLOR_MAP,
    DEFAULT_FALLBACK_CMAP,
    UNIVERSAL_FONTSIZE
)
# ---

# ---
# ---  SPATIOTEMPORAL CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- File Paths ---
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv",
    "output_savename": "sunburst_st_distribution_v3_ring.svg", 
    
    # --- Sunburst Chart Styling ---
    "hole_radius": 0.0, # <-- SET TO 0.0 TO REMOVE DONUT HOLE
    "l1_ring_radius": 0.5,  # Innermost ring (L1)
    "l2_ring_radius": 0.9,  # Middle ring (L2)
    "l3_ring_radius": 1.4,  # Outermost ring (L3)
    
    # --- Color Gradient Ranges ---
    "l2_gradient_range": [0.3, 0.6], # L2 will be medium-light
    "l3_gradient_range": [0.7, 1.0]  # L3 will be lightest
}
# ---
# --- END CONTROL PANEL ---
# ---

# (ST_HIERARCHY_TEXT is unchanged)
ST_HIERARCHY_TEXT = """
    # 1.0 Spatial Relationships (Static): Relationships inferable from a single moment.
        # 1.1 Relative Position (e.g., Near, Beside, InFrontOf, Behind, Above, Below, Between)
        # 1.2 Contact: Interactions involving direct physical touch.
            # 1.2.1 Supportive Contact (e.g., SitsOn, LeansOn, StandsOn, LiesOn, RestsOn)
            # 1.2.2 Manipulative Contact (e.g., Holds, Grabs, Wears, Carries, Uses)
            # 1.2.3 Social/Affectionate Contact (e.g., Hugs, Kisses, HoldsHand, High-fives)
        # 1.3 Perceptual & Indicative Relationships: Non-physical links based on senses or gestures.
            # 1.3.1 Gaze (e.g., Watching, LookingAt, StaresAt, GlancesAt)
            # 1.3.2 Indicative Gesture (e.g., PointsTo, GesturesTowards, NodsAt)
        # 1.4 Communicative Acts (e.g., SpeaksTo, ListensTo, Nods, ShakesHead)
    # 2.0 State Changes & Sequential Actions (Temporal): Relationships defined by change or duration over time, without significant change in relative spatial position.
        # 2.1 Actor State Change (e.g., StandsUp, SitsDown, TurnsAround, BendsOver, UnfastensSeatbelt, Nods, ShakesHead)
        # 2.2 Object State Change (e.g., OpensDoor, ClosesDoor, TurnsOnLight, LightsCigarette)
        # 2.3 Sequential Actions (e.g., Knocks then HandsOver, Bends then CoversMouth, Unfastens then StandsUp)
        # 2.4 Durational States & Non-Actions (e.g., Waits, Pauses, Hesitates, Sleeps, StandsStill, RemainsSeated, SpeaksTo, ListensTo)
    # 3.0 Spatio-Temporal Interactions (Composite): Combines movement through space with a changing relationship.
        # 3.1 Relative Motion: An entity's movement described in relation to another.
            # 3.1.1 Approach & Depart (e.g., WalksTowards, RunsTo, MovesAwayFrom, BacksAwayFrom)
            # 3.1.2 Passing & Crossing (e.g., WalksPast, CreepsPast, FliesNextTo, JumpsBeneath, DrivesAlongside)
            # 3.1.3 Following & Leading (e.g., Follows, Chases, Leads, Escorts)
        # 3.2 Object Transference (e.g., HandsTo, Gives, PicksUp, PutsDown, TakesOff)
        # 3.3 Instantaneous Motion & Impact (e.g., Hits, Touches, Taps, Kicks, Pushes, Throws, Drops)
        # 3.4 Composite Action Sequences (e.g., WalksTo then SitsOn, ClosesDoor and JumpsIn, TurnsHead and SpeaksTo)
"""
# (parse_st_hierarchy function is unchanged)
def parse_st_hierarchy(hierarchy_text):
    l1_map = {}
    l2_map = {}
    l3_map = {}
    current_l1_id = None
    current_l2_id = None
    for line in hierarchy_text.strip().split('\n'):
        line = line.strip().lstrip('#').strip()
        if not line:
            continue
        l3_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l3_match and current_l2_id:
            l3_id = l3_match.group(1)
            l3_name = f"{l3_id} {l3_match.group(2)}"
            if l3_id.startswith(current_l2_id):
                l3_map[l3_id] = l3_name
            else:
                print(f"Warning: L3 ID {l3_id} does not match L2 parent {current_l2_id}")
            continue
        l1_match = re.match(r'^(\d+)\.0\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l1_match:
            l1_id = l1_match.group(1)
            l1_name = f"{l1_id}.0 {l1_match.group(2).rstrip(':')}"
            l1_map[l1_id] = l1_name
            current_l1_id = l1_id
            current_l2_id = None
            continue
        l2_match = re.match(r'^(\d+\.\d+)\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l2_match and current_l1_id:
            l2_id = l2_match.group(1)
            l2_name = f"{l2_id} {l2_match.group(2).rstrip(':')}"
            if l2_id.startswith(current_l1_id + '.'):
                l2_map[l2_id] = l2_name
                current_l2_id = l2_id
            else:
                print(f"Warning: L2 ID {l2_id} does not match L1 parent {current_l1_id}")
            continue
    return l1_map, l2_map, l3_map


def create_st_sunburst_chart(savename, df, 
                             base_cmaps,
                             coarse_color_map,
                             default_cmap_obj,
                             hole_radius, 
                             l1_ring_radius, 
                             l2_ring_radius, 
                             l3_ring_radius,
                             l2_gradient_range,
                             l3_gradient_range,
                             universal_fontsize):
    """
    Creates a 3-level, static sunburst chart using Matplotlib.
    """
    
    # 1. --- Data Preparation --- (Unchanged)
    df_l1 = df.groupby('level1_cat').size().reset_index(name='counts')
    df_l1 = df_l1.sort_values('level1_cat').reset_index(drop=True)
    # df_l1['label'] = df_l1['level1_cat'].apply(lambda x: x.split(" ")[0])
    df_l1['label'] = df_l1['level1_cat'].apply(lambda x: x[:15])
    
    df_l2 = df.groupby(['level1_cat', 'level2_cat']).size().reset_index(name='counts')
    df_l2 = df_l2.sort_values(['level1_cat', 'level2_cat']).reset_index(drop=True)
    # df_l2['label'] = df_l2['level2_cat'].apply(lambda x: x.split(" ")[0])
    df_l2['label'] = df_l2['level2_cat'].apply(lambda x: x[:15])
    
    df_l3 = df.groupby(['level1_cat', 'level2_cat', 'finest_cat']).size().reset_index(name='counts')
    df_l3 = df_l3.sort_values(['level1_cat', 'level2_cat', 'finest_cat']).reset_index(drop=True)
    # df_l3['label'] = df_l3['finest_cat'].apply(lambda x: x.split(" ")[0])
    df_l3['label'] = df_l3['finest_cat'].apply(lambda x: x[:15])
    


    # 2. --- Color Generation --- (Unchanged)
    
    # --- L1 Ring Colors (Solid) ---
    l1_colors = df_l1['level1_cat'].map(coarse_color_map)
    if l1_colors.isna().any():
        missing = df_l1[l1_colors.isna()]['level1_cat'].unique()
        raise AssertionError(f"Missing L1 color keys: {missing}. Check keys in `coarse_color_map`.")

    # --- L2 Ring Colors (Gradient from L1) ---
    l2_color_map = {}
    unique_l1_categories = df_l1['level1_cat'].unique()
    for l1_cat in unique_l1_categories:
        cmap = base_cmaps.get(l1_cat, default_cmap_obj)
        l2_children = df_l2[df_l2['level1_cat'] == l1_cat]['level2_cat'].unique()
        num_l2_children = len(l2_children)
        if num_l2_children == 0: continue
        l2_palette = [cmap(i) for i in np.linspace(l2_gradient_range[0], l2_gradient_range[1], num_l2_children)]
        l2_color_map.update(zip(l2_children, l2_palette))
    l2_colors = df_l2['level2_cat'].map(l2_color_map)
    if l2_colors.isna().any():
        l2_colors = l2_colors.fillna('#808080')

    # --- L3 Ring Colors (Gradient from L1, but lighter) ---
    l3_color_map = {}
    for l1_cat in unique_l1_categories:
        cmap = base_cmaps.get(l1_cat, default_cmap_obj)
        l3_children = df_l3[df_l3['level1_cat'] == l1_cat]['finest_cat'].unique()
        num_l3_children = len(l3_children)
        if num_l3_children == 0: continue
        l3_palette = [cmap(i) for i in np.linspace(l3_gradient_range[0], l3_gradient_range[1], num_l3_children)]
        l3_color_map.update(zip(l3_children, l3_palette))
    for l2_cat in df_l2['level2_cat'].unique():
        if l2_cat in l3_color_map and l2_cat in l2_color_map:
             l3_color_map[l2_cat] = l2_color_map[l2_cat]
    l3_colors = df_l3['finest_cat'].map(l3_color_map)
    if l3_colors.isna().any():
        l3_colors = l3_colors.fillna('#808080')
        
    # --- MODIFICATION: Make L2-leaf slices transparent in L3 ring ---
    is_l2_leaf = (df_l3['level2_cat'] == df_l3['finest_cat'])
    num_leaves_to_set = is_l2_leaf.sum()
    if num_leaves_to_set > 0:
        transparent_list = [(0, 0, 0, 0)] * num_leaves_to_set
        l3_colors.loc[is_l2_leaf] = transparent_list # Set facecolor to transparent
    
    # Also create a list of edgecolors for L3
    l3_edgecolors = ['w'] * len(df_l3)
    for i, is_leaf in enumerate(is_l2_leaf): 
        if is_leaf:
            l3_edgecolors[i] = (0, 0, 0, 0) # Set edgecolor to transparent


    # 3. --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(aspect="equal"))
    plt.style.use('default') 

    # Base properties for L1 and L2
    base_wedge_props = dict(edgecolor='w', linewidth=1.0)
    text_props = dict(fontweight='normal', ha='center', va='center')
    
    MIN_FONT_SIZE = 8 
    REFERENCE_ANGLE = 5 
    
    # --- Draw Outer Ring (L3) ---
    # --- FIX: Removed edgecolor from wedgeprops ---
    outer_wedges, _ = ax.pie(
        df_l3['counts'],
        radius=l3_ring_radius,
        colors=l3_colors.tolist(), # Pass the modified colors
        wedgeprops=dict(width=l3_ring_radius - l2_ring_radius, 
                          # edgecolor removed from here
                          linewidth=1.0), 
        startangle=90,
    )
    # --- FIX: Manually set edge colors for each L3 wedge ---
    for i, wedge in enumerate(outer_wedges):
        wedge.set_edgecolor(l3_edgecolors[i])

    # --- Draw Middle Ring (L2) ---
    middle_wedges, _ = ax.pie(
        df_l2['counts'],
        radius=l2_ring_radius,
        colors=l2_colors.tolist(),
        wedgeprops=dict(width=l2_ring_radius - l1_ring_radius, **base_wedge_props), 
        startangle=90,
    )

    # --- Draw Inner Ring (L1) ---
    inner_wedges, _ = ax.pie(
        df_l1['counts'],
        radius=l1_ring_radius,
        colors=l1_colors.tolist(),
        wedgeprops=dict(width=l1_ring_radius - hole_radius, **base_wedge_props), 
        startangle=90,
    )

    # 4. --- Labeling --- (Unchanged)
    
    # Add labels for the inner ring (L1)
    for i, p in enumerate(inner_wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1 
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = hole_radius + (l1_ring_radius - hole_radius) / 2
        ax.text(x * text_radius, y * text_radius,
                df_l1['label'][i], rotation=rotation, color='black',
                fontsize=universal_fontsize, **text_props)

    # Add labels for the middle ring (L2)
    for i, p in enumerate(middle_wedges):
        slice_angle = p.theta2 - p.theta1
        if slice_angle < 1.0: continue 
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = l1_ring_radius + (l2_ring_radius - l1_ring_radius) / 2
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
        ax.text(x * text_radius, y * text_radius,
                df_l2['label'][i], rotation=rotation, color='black',
                fontsize=dynamic_fontsize, **text_props)
                
    # Add labels for the outer ring (L3)
    for i, p in enumerate(outer_wedges):
        slice_angle = p.theta2 - p.theta1
        if slice_angle < 1.0: continue 
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = l2_ring_radius + (l3_ring_radius - l2_ring_radius) / 2
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
        if df_l3['finest_cat'][i] == df_l3['level2_cat'][i]:
            continue
        ax.text(x * text_radius, y * text_radius,
                df_l3['label'][i], rotation=rotation, color='black',
                fontsize=dynamic_fontsize, **text_props)
    
    # 5. --- Save Static Image --- (Unchanged)
    print(f"Saving static chart to {savename}...")
    fig.savefig(savename, format='svg', bbox_inches='tight')
    fig.tight_layout()
    plt.show()
    print("Done.")


def main():
    """
    Main execution function to load data, generate colors, and create the chart.
    """
    
    # --- 1. Setup Environment --- (Unchanged)
    print("Running in 'spatiotemporal' (st) mode.")
    data_path = CONTROL_PANEL["st_data_path"]
    
    print("Parsing ST hierarchy...")
    L1_MAP, L2_MAP, L3_MAP = parse_st_hierarchy(ST_HIERARCHY_TEXT)
    print(f"Found {len(L1_MAP)} L1, {len(L2_MAP)} L2, and {len(L3_MAP)} L3 categories.")
    
    plot_coarse_color_map = {}
    plot_base_cmaps = {}
    
    for l1_id_str, display_name in L1_MAP.items():
        base_name = ST_COARSE_CATEGORIES_MAP[int(l1_id_str)]
        if base_name in ST_COARSE_COLOR_MAP:
            plot_coarse_color_map[display_name] = ST_COARSE_COLOR_MAP[base_name]
        if base_name in ST_BASE_COLORMAPS:
            plot_base_cmaps[display_name] = ST_BASE_COLORMAPS[base_name]

    # --- 2. Load Data --- (Unchanged)
    print(f"Loading classification data from {data_path}")
    try:
        classification_df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    print(f"Loaded {len(classification_df)} total annotations.")

    # --- 3. Process Data into DataFrame --- (Unchanged)
    processed_records = [] 
    skip_counters = {
        'missing_category': 0,
        'malformed_id': 0,
        'unknown_id': 0
    }
    print("Processing annotations...")
    for _, row in classification_df.iterrows():
        st_cat_str = row.get('category')
        if not st_cat_str or pd.isna(st_cat_str):
            skip_counters['missing_category'] += 1
            continue
        full_id_str = str(st_cat_str).split(" ")[0]
        id_parts = full_id_str.split('.')
        if len(id_parts) < 2:
            skip_counters['malformed_id'] += 1
            continue
        l1_id = id_parts[0]
        l2_id = f"{id_parts[0]}.{id_parts[1]}"
        l3_id = full_id_str if len(id_parts) > 2 else None
        l1_name = L1_MAP.get(l1_id)
        l2_name = L2_MAP.get(l2_id)
        l3_name = L3_MAP.get(l3_id)
        if not l1_name or not l2_name:
            skip_counters['unknown_id'] += 1
            continue
        finest_cat_name = l3_name if l3_name else l2_name
        processed_records.append({
            "level1_cat": l1_name,
            "level2_cat": l2_name,
            "level3_cat": l3_name,
            "finest_cat": finest_cat_name,
        })
    df = pd.DataFrame(processed_records)
    print(f"Finished processing. Skipped annotations summary:")
    print(f"  - Missing category string: {skip_counters['missing_category']}")
    print(f"  - Malformed ID (e.g., < 2 parts): {skip_counters['malformed_id']}")
    print(f"  - Unknown ID (not in hierarchy): {skip_counters['unknown_id']}")
    if df.empty:
        print("No valid data processed. Exiting.")
        return
    print(f'Processed {len(df)} valid hierarchical annotations.')

    # --- 4. Create and Save Chart ---
    create_st_sunburst_chart(
        savename=CONTROL_PANEL['output_savename'], 
        df=df, 
        base_cmaps=plot_base_cmaps,
        coarse_color_map=plot_coarse_color_map,
        default_cmap_obj=DEFAULT_FALLBACK_CMAP,
        hole_radius=CONTROL_PANEL['hole_radius'], # <-- Will pass 0.0
        l1_ring_radius=CONTROL_PANEL['l1_ring_radius'], 
        l2_ring_radius=CONTROL_PANEL['l2_ring_radius'],
        l3_ring_radius=CONTROL_PANEL['l3_ring_radius'],
        l2_gradient_range=CONTROL_PANEL['l2_gradient_range'],
        l3_gradient_range=CONTROL_PANEL['l3_gradient_range'],
        universal_fontsize=UNIVERSAL_FONTSIZE
    )

if __name__ == "__main__":
    main()
# %%
# %%
# --- NEW CELL FOR EQUAL-DISTRIBUTION PLOT ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.colors as mcolors

# --- Import constants (assuming they are in scope) ---
from constants import (
    ST_COARSE_CATEGORIES_MAP,
    ST_BASE_COLORMAPS,
    ST_COARSE_COLOR_MAP,
    DEFAULT_FALLBACK_CMAP,
    UNIVERSAL_FONTSIZE
)

# ---
# ---  EQUAL-DISTRIBUTION CONTROL PANEL  ---
# ---
EQUAL_DIST_CONTROL_PANEL = {
    "output_savename": "sunburst_st_equal_dist.svg", 
    
    # --- Sunburst Chart Styling ---
    "hole_radius": 0.0,
    "l1_ring_radius": 0.5,
    "l2_ring_radius": 0.9,
    "l3_ring_radius": 1.4,
    
    # --- Color Gradient Ranges ---
    "l2_gradient_range": [0.3, 0.6],
    "l3_gradient_range": [0.7, 1.0]
}
# ---
# --- END CONTROL PANEL ---
# ---

# (ST_HIERARCHY_TEXT is unchanged)
ST_HIERARCHY_TEXT = """
    # 1.0 Spatial Relationships (Static): Relationships inferable from a single moment.
        # 1.1 Relative Position (e.g., Near, Beside, InFrontOf, Behind, Above, Below, Between)
        # 1.2 Contact: Interactions involving direct physical touch.
            # 1.2.1 Supportive Contact (e.g., SitsOn, LeansOn, StandsOn, LiesOn, RestsOn)
            # 1.2.2 Manipulative Contact (e.g., Holds, Grabs, Wears, Carries, Uses)
            # 1.2.3 Social/Affectionate Contact (e.g., Hugs, Kisses, HoldsHand, High-fives)
        # 1.3 Perceptual & Indicative Relationships: Non-physical links based on senses or gestures.
            # 1.3.1 Gaze (e.g., Watching, LookingAt, StaresAt, GlancesAt)
            # 1.3.2 Indicative Gesture (e.g., PointsTo, GesturesTowards, NodsAt)
        # 1.4 Communicative Acts (e.g., SpeaksTo, ListensTo, Nods, ShakesHead)

    # 2.0 State Changes & Sequential Actions (Temporal): Relationships defined by change or duration over time, without significant change in relative spatial position.
        # 2.1 Actor State Change (e.g., StandsUp, SitsDown, TurnsAround, BendsOver, UnfastensSeatbelt, Nods, ShakesHead)
        # 2.2 Object State Change (e.g., OpensDoor, ClosesDoor, TurnsOnLight, LightsCigarette)
        # 2.3 Sequential Actions (e.g., Knocks then HandsOver, Bends then CoversMouth, Unfastens then StandsUp)
        # 2.4 Durational States & Non-Actions (e.g., Waits, Pauses, Hesitates, Sleeps, StandsStill, RemainsSeated, SpeaksTo, ListensTo)

    # 3.0 Spatio-Temporal Interactions (Composite): Combines movement through space with a changing relationship.
        # 3.1 Relative Motion: An entity's movement described in relation to another.
            # 3.1.1 Approach & Depart (e.g., WalksTowards, RunsTo, MovesAwayFrom, BacksAwayFrom)
            # 3.1.2 Passing & Crossing (e.g., WalksPast, CreepsPast, FliesNextTo, JumpsBeneath, DrivesAlongside)
            # 3.1.3 Following & Leading (e.g., Follows, Chases, Leads, Escorts)
        # 3.2 Object Transference (e.g., HandsTo, Gives, PicksUp, PutsDown, TakesOff)
        # 3.3 Instantaneous Motion & Impact (e.g., Hits, Touches, Taps, Kicks, Pushes, Throws, Drops)
        # 3.4 Composite Action Sequences (e.g., WalksTo then SitsOn, ClosesDoor and JumpsIn, TurnsHead and SpeaksTo)
"""
# (parse_st_hierarchy function is unchanged)
def parse_st_hierarchy(hierarchy_text):
    l1_map = {}
    l2_map = {}
    l3_map = {}
    current_l1_id = None
    current_l2_id = None
    for line in hierarchy_text.strip().split('\n'):
        line = line.strip().lstrip('#').strip()
        if not line:
            continue
        l3_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l3_match and current_l2_id:
            l3_id = l3_match.group(1)
            l3_name = f"{l3_id} {l3_match.group(2)}"
            if l3_id.startswith(current_l2_id):
                l3_map[l3_id] = l3_name
            else:
                print(f"Warning: L3 ID {l3_id} does not match L2 parent {current_l2_id}")
            continue
        l1_match = re.match(r'^(\d+)\.0\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l1_match:
            l1_id = l1_match.group(1)
            l1_name = f"{l1_id}.0 {l1_match.group(2).rstrip(':')}"
            l1_map[l1_id] = l1_name
            current_l1_id = l1_id
            current_l2_id = None
            continue
        l2_match = re.match(r'^(\d+\.\d+)\s+(.*?)(\s+\(e\.g\.,.*)?$', line)
        if l2_match and current_l1_id:
            l2_id = l2_match.group(1)
            l2_name = f"{l2_id} {l2_match.group(2).rstrip(':')}"
            if l2_id.startswith(current_l1_id + '.'):
                l2_map[l2_id] = l2_name
                current_l2_id = l2_id
            else:
                print(f"Warning: L2 ID {l2_id} does not match L1 parent {current_l1_id}")
            continue
    return l1_map, l2_map, l3_map


def generate_equal_dist_finest_df(l1_map, l2_map, l3_map):
    """
    Generates a 'finest_level' dataframe where counts are
    distributed equally at each level of the hierarchy.
    """
    processed_records = []
    
    # Find all L1 categories
    l1_names = l1_map.values()
    if not l1_names:
        return pd.DataFrame()
        
    l1_base_count = 1.0 / len(l1_names)

    for l1_name in l1_names:
        l1_id_prefix = l1_name.split(" ")[0].split(".")[0] + "."
        
        # Find all L2 children for this L1
        l2_children = [name for id, name in l2_map.items() if id.startswith(l1_id_prefix)]
        if not l2_children:
            continue
            
        l2_base_count = l1_base_count / len(l2_children)
        
        for l2_name in l2_children:
            l2_id_prefix = l2_name.split(" ")[0]
            
            # Find all L3 children for this L2
            l3_children = [name for id, name in l3_map.items() if id.startswith(l2_id_prefix + ".")]
            
            if not l3_children:
                # This is an L2-leaf node
                processed_records.append({
                    "level1_cat": l1_name,
                    "level2_cat": l2_name,
                    "level3_cat": None,
                    "finest_cat": l2_name,
                    "counts": l2_base_count # This leaf gets the full L2 count
                })
            else:
                # This L2 node has L3 children
                l3_base_count = l2_base_count / len(l3_children)
                for l3_name in l3_children:
                    processed_records.append({
                        "level1_cat": l1_name,
                        "level2_cat": l2_name,
                        "level3_cat": l3_name,
                        "finest_cat": l3_name,
                        "counts": l3_base_count # Each L3 gets its fraction
                    })

    return pd.DataFrame(processed_records)


def create_st_sunburst_chart_equal_dist(savename, df_finest, 
                                        base_cmaps,
                                        coarse_color_map,
                                        default_cmap_obj,
                                        hole_radius, 
                                        l1_ring_radius, 
                                        l2_ring_radius, 
                                        l3_ring_radius,
                                        l2_gradient_range,
                                        l3_gradient_range,
                                        universal_fontsize):
    """
    Creates a 3-level, static sunburst chart using Matplotlib
    from a dataframe with pre-computed 'counts'.
    """
    
    # 1. --- Data Preparation ---
    # --- MODIFICATION: Use .sum() on 'counts' instead of .size() ---
    
    # df_finest is the L3 dataframe
    df_l3 = df_finest.sort_values(['level1_cat', 'level2_cat', 'finest_cat']).reset_index(drop=True)
    df_l3['label'] = df_l3['finest_cat'].apply(lambda x: x[:15])
    
    # Aggregate counts up to L2
    df_l2 = df_l3.groupby(['level1_cat', 'level2_cat'])['counts'].sum().reset_index()
    df_l2 = df_l2.sort_values(['level1_cat', 'level2_cat']).reset_index(drop=True)
    df_l2['label'] = df_l2['level2_cat'].apply(lambda x: x[:15])
    
    # Aggregate counts up to L1
    df_l1 = df_l2.groupby('level1_cat')['counts'].sum().reset_index()
    df_l1 = df_l1.sort_values('level1_cat').reset_index(drop=True)
    df_l1['label'] = df_l1['level1_cat'].apply(lambda x: x[:15])


    # 2. --- Color Generation --- (Unchanged)
    
    # --- L1 Ring Colors (Solid) ---
    l1_colors = df_l1['level1_cat'].map(coarse_color_map)
    if l1_colors.isna().any():
        missing = df_l1[l1_colors.isna()]['level1_cat'].unique()
        raise AssertionError(f"Missing L1 color keys: {missing}. Check keys in `coarse_color_map`.")

    # --- L2 Ring Colors (Gradient from L1) ---
    l2_color_map = {}
    unique_l1_categories = df_l1['level1_cat'].unique()
    for l1_cat in unique_l1_categories:
        cmap = base_cmaps.get(l1_cat, default_cmap_obj)
        l2_children = df_l2[df_l2['level1_cat'] == l1_cat]['level2_cat'].unique()
        num_l2_children = len(l2_children)
        if num_l2_children == 0: continue
        l2_palette = [cmap(i) for i in np.linspace(l2_gradient_range[0], l2_gradient_range[1], num_l2_children)]
        l2_color_map.update(zip(l2_children, l2_palette))
    l2_colors = df_l2['level2_cat'].map(l2_color_map)
    if l2_colors.isna().any():
        l2_colors = l2_colors.fillna('#808080')

    # --- L3 Ring Colors (Gradient from L1, but lighter) ---
    l3_color_map = {}
    for l1_cat in unique_l1_categories:
        cmap = base_cmaps.get(l1_cat, default_cmap_obj)
        l3_children = df_l3[df_l3['level1_cat'] == l1_cat]['finest_cat'].unique()
        num_l3_children = len(l3_children)
        if num_l3_children == 0: continue
        l3_palette = [cmap(i) for i in np.linspace(l3_gradient_range[0], l3_gradient_range[1], num_l3_children)]
        l3_color_map.update(zip(l3_children, l3_palette))
    for l2_cat in df_l2['level2_cat'].unique():
        if l2_cat in l3_color_map and l2_cat in l2_color_map:
             l3_color_map[l2_cat] = l2_color_map[l2_cat]
    l3_colors = df_l3['finest_cat'].map(l3_color_map)
    if l3_colors.isna().any():
        l3_colors = l3_colors.fillna('#808080')
        
    # --- MODIFICATION: Make L2-leaf slices transparent in L3 ring ---
    is_l2_leaf = (df_l3['level2_cat'] == df_l3['finest_cat'])
    num_leaves_to_set = is_l2_leaf.sum()
    if num_leaves_to_set > 0:
        transparent_list = [(0, 0, 0, 0)] * num_leaves_to_set
        l3_colors.loc[is_l2_leaf] = transparent_list
    
    l3_edgecolors = ['w'] * len(df_l3)
    for i, is_leaf in enumerate(is_l2_leaf): 
        if is_leaf:
            l3_edgecolors[i] = (0, 0, 0, 0)


    # 3. --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(aspect="equal"))
    plt.style.use('default') 

    base_wedge_props = dict(edgecolor='w', linewidth=1.0)
    text_props = dict(fontweight='normal', ha='center', va='center')
    
    MIN_FONT_SIZE = 8 
    REFERENCE_ANGLE = 5 
    
    # --- Draw Outer Ring (L3) ---
    outer_wedges, _ = ax.pie(
        df_l3['counts'], # <-- Use 'counts' column
        radius=l3_ring_radius,
        colors=l3_colors.tolist(),
        wedgeprops=dict(width=l3_ring_radius - l2_ring_radius, 
                          linewidth=1.0), 
        startangle=90,
    )
    for i, wedge in enumerate(outer_wedges):
        wedge.set_edgecolor(l3_edgecolors[i])

    # --- Draw Middle Ring (L2) ---
    middle_wedges, _ = ax.pie(
        df_l2['counts'], # <-- Use 'counts' column
        radius=l2_ring_radius,
        colors=l2_colors.tolist(),
        wedgeprops=dict(width=l2_ring_radius - l1_ring_radius, **base_wedge_props), 
        startangle=90,
    )

    # --- Draw Inner Ring (L1) ---
    inner_wedges, _ = ax.pie(
        df_l1['counts'], # <-- Use 'counts' column
        radius=l1_ring_radius,
        colors=l1_colors.tolist(),
        wedgeprops=dict(width=l1_ring_radius - hole_radius, **base_wedge_props), 
        startangle=90,
    )

    # 4. --- Labeling --- (Unchanged)
    for i, p in enumerate(inner_wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1 
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = hole_radius + (l1_ring_radius - hole_radius) / 2
        ax.text(x * text_radius, y * text_radius,
                df_l1['label'][i], rotation=rotation, color='black',
                fontsize=universal_fontsize, **text_props)
    for i, p in enumerate(middle_wedges):
        slice_angle = p.theta2 - p.theta1
        if slice_angle < 1.0: continue 
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = l1_ring_radius + (l2_ring_radius - l1_ring_radius) / 2
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
        ax.text(x * text_radius, y * text_radius,
                df_l2['label'][i], rotation=rotation, color='black',
                fontsize=dynamic_fontsize, **text_props)
    for i, p in enumerate(outer_wedges):
        slice_angle = p.theta2 - p.theta1
        if slice_angle < 1.0: continue 
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        rotation = ang
        if 90 < ang < 270: rotation = ang + 180
        text_radius = l2_ring_radius + (l3_ring_radius - l2_ring_radius) / 2
        scaled_fontsize = universal_fontsize * (slice_angle / REFERENCE_ANGLE)
        dynamic_fontsize = max(MIN_FONT_SIZE, min(scaled_fontsize, universal_fontsize))
        if df_l3['finest_cat'][i] == df_l3['level2_cat'][i]:
            continue
        ax.text(x * text_radius, y * text_radius,
                df_l3['label'][i], rotation=rotation, color='black',
                fontsize=dynamic_fontsize, **text_props)
    
    # 5. --- Save Static Image --- (Unchanged)
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
    print("Running in 'spatiotemporal' (st) EQUAL mode.")
    
    print("Parsing ST hierarchy...")
    L1_MAP, L2_MAP, L3_MAP = parse_st_hierarchy(ST_HIERARCHY_TEXT)
    print(f"Found {len(L1_MAP)} L1, {len(L2_MAP)} L2, and {len(L3_MAP)} L3 categories.")
    
    # Re-key the color maps (same as before)
    plot_coarse_color_map = {}
    plot_base_cmaps = {}
    for l1_id_str, display_name in L1_MAP.items():
        base_name = ST_COARSE_CATEGORIES_MAP[int(l1_id_str)]
        if base_name in ST_COARSE_COLOR_MAP:
            plot_coarse_color_map[display_name] = ST_COARSE_COLOR_MAP[base_name]
        if base_name in ST_BASE_COLORMAPS:
            plot_base_cmaps[display_name] = ST_BASE_COLORMAPS[base_name]

    # --- 2. Generate Data ---
    print("Generating equal distribution data...")
    df_finest = generate_equal_dist_finest_df(L1_MAP, L2_MAP, L3_MAP)
    
    if df_finest.empty:
        print("No data generated. Exiting.")
        return
        
    print(f'Generated {len(df_finest)} finest-level records with equal counts.')

    # --- 3. Create and Save Chart ---
    create_st_sunburst_chart_equal_dist(
        savename=EQUAL_DIST_CONTROL_PANEL['output_savename'], 
        df_finest=df_finest, 
        base_cmaps=plot_base_cmaps,
        coarse_color_map=plot_coarse_color_map,
        default_cmap_obj=DEFAULT_FALLBACK_CMAP,
        hole_radius=EQUAL_DIST_CONTROL_PANEL['hole_radius'],
        l1_ring_radius=EQUAL_DIST_CONTROL_PANEL['l1_ring_radius'], 
        l2_ring_radius=EQUAL_DIST_CONTROL_PANEL['l2_ring_radius'],
        l3_ring_radius=EQUAL_DIST_CONTROL_PANEL['l3_ring_radius'],
        l2_gradient_range=EQUAL_DIST_CONTROL_PANEL['l2_gradient_range'],
        l3_gradient_range=EQUAL_DIST_CONTROL_PANEL['l3_gradient_range'],
        universal_fontsize=UNIVERSAL_FONTSIZE
    )

if __name__ == "__main__":
    print("--- Running Equal Distribution Plot ---")
    main_equal_dist()
# %%