# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from constants import (
    ST_COARSE_COLOR_MAP, ENTITY_COARSE_COLOR_MAP,
    UNIVERSAL_FONTSIZE # <-- IMPORT THE FONTSIZE
)

df = pd.read_csv('/home/aparcedo/IASEB/interaction_analysis/alejandro_dalton_anirudh_table1_categorized.csv')

# %%

# Define a default fontsize if not imported

# %%
def create_and_save_radar_chart_with_ring(df, 
                                          coarse_color_map, 
                                          save_path=None, 
                                          category_col='fine_category', 
                                          coarse_category_col='coarse_category',
                                          category_sort_col=None,  # <-- For fine-grained sorting key
                                          coarse_sort_col=None,   # <-- For coarse sorting key
                                          color_by_model=True,
                                          universal_fontsize=UNIVERSAL_FONTSIZE):
    """
    Generates a radar chart with an outer ring, sorted by the
    provided key columns.
    """
    
    # --- START OF MODIFICATIONS ---
    
    # If no sort keys are provided, fall back to sorting by the label/color columns
    sort_col_fine = category_sort_col if category_sort_col else category_col
    sort_col_coarse = coarse_sort_col if coarse_sort_col else coarse_category_col
    
    # Define all columns we will need for this plot
    required_cols = [category_col, coarse_category_col, sort_col_fine, sort_col_coarse]
    
    # Check that they all exist
    for col in set(required_cols): # Use set() to avoid duplicate checks
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in DataFrame.")
            return

    # Drop rows where *any* of our needed columns are NaN
    plot_df = df.dropna(subset=list(set(required_cols)))
    if plot_df.empty: 
        print(f"Skipping chart for '{category_col}': No data after dropping NaN."); 
        return

    # --- THIS IS THE NEW SORTING LOGIC ---
    # 1. Get a DataFrame of all unique category/label/key combinations
    cols_to_get = [coarse_category_col, category_col, sort_col_coarse, sort_col_fine]
    category_df = plot_df[cols_to_get].drop_duplicates()
    
    # 2. Sort this DataFrame by the KEY columns
    #    Note: This assumes the key columns can be sorted correctly (e.g., numbers or zero-padded strings)
    #    If keys are "1.1", "1.10", "1.2", they will sort alphabetically.
    #    If this is a problem, the keys may need to be converted to a sortable type.
    #    For now, we assume simple numeric or dot-separated string sorting works.
    category_df = category_df.sort_values(by=[sort_col_coarse, sort_col_fine])
    
    # 3. Get the sorted list of *labels* from the category_col
    categories = category_df[category_col].tolist()
    # --- END OF SORTING LOGIC ---

    models = plot_df['model'].unique(); models.sort()
    
    if len(categories) < 3: 
        print(f"Skipping chart for '{category_col}': Not enough categories."); 
        return
    
    # --- END OF MODIFICATIONS ---

    # The rest of the function is identical
    pivot_df = plot_df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0) # This now uses the sorted list

    # Create map of fine-category-label -> coarse-category-label
    coarse_map = pd.Series(plot_df[coarse_category_col].values, index=plot_df[category_col]).to_dict()
    
    # Get the list of coarse category labels, IN THE NEW SORTED ORDER
    sorted_categories = list(pivot_df.columns) # This is the fine-category labels, sorted
    try: 
        # This list will contain coarse labels, e.g., ['Spatial', 'Spatial', 'Temporal', ...]
        sorted_coarse_categories = [coarse_map[cat] for cat in sorted_categories]
    except KeyError as e: 
        print(f"Error: Could not find coarse category for fine category '{e.key}'."); 
        return

    max_iou = pivot_df.values.max(); data_max = max_iou + 0.05
    if data_max == 0.05: data_max = 1.0

    num_vars = len(categories)
    
    # Angles for the data lines
    angles_plot = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_plot += angles_plot[:1] 
    
    # Angles for labels and bars
    angles_labels = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    
    figsize = (12, 12) 
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist(); values += values[:1]
        color_map = plt.cm.get_cmap('tab20'); color = color_map(i % color_map.N) if color_by_model else 'blue'
        ax.plot(angles_plot, values, color=color, linewidth=2, label=model, zorder=2)
        ax.fill(angles_plot, values, color=color, alpha=0.25, zorder=2)

    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_rlabel_position(0)
    
    ax.set_xticks(angles_labels)
    ax.set_xticklabels([]) 

    arc_inner_radius = data_max
    arc_thickness_ratio = 0.08 
    arc_outer_radius = data_max * (1 + arc_thickness_ratio)
    
    segment_width = 2 * np.pi / num_vars
    start_angles = angles_labels

    for i in range(num_vars):
        coarse_cat = sorted_coarse_categories[i] # This is now the coarse label, e.g., 'Spatial (Static)'
        color = coarse_color_map.get(coarse_cat, '#808080') # This lookup now works
        ax.bar(
            x=start_angles[i],
            height=arc_outer_radius - arc_inner_radius,
            width=segment_width,
            bottom=arc_inner_radius,
            color=color,
            alpha=0.7,
            zorder=1,
            align='edge'
        )

    ax.spines['polar'].set_visible(False)
    
    label_radius = arc_inner_radius * 1.1
    
    for i, category in enumerate(categories):
        angle_rad = angles_labels[i] 
        ax.text(angle_rad, 
                label_radius,
                category,
                fontsize=universal_fontsize,
                fontweight='light',
                rotation=0,
                ha='center',   
                va='center',  
                color='black',
                zorder=3)
    
    ax.set_yticks(np.arange(0, data_max, data_max / 5))
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, data_max, data_max / 5)], color="grey", size=universal_fontsize)
    
    final_chart_max = arc_outer_radius * 1.0
    ax.set_ylim(0, final_chart_max)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.85), fontsize=universal_fontsize, frameon=False)
    # fig.tight_layout(rect=[0, 0, 0.8, 1])

    if save_path:
        fig.savefig(save_path, format='png', bbox_inches='tight'); plt.close(fig)
        print(f"Radar chart saved to {save_path}")
    else: 
        plt.show()
# %%
create_and_save_radar_chart_with_ring(
        df, 
        coarse_color_map=ST_COARSE_COLOR_MAP, 
        save_path='st_l2_radar_chart_20251109_2115.png', 
        
        # Columns for labels and colors
        category_col='st_level1_cls',
        coarse_category_col='st_level0_cls', 
        
        # NEW: Columns for sorting
        category_sort_col='st_level1_key',
        coarse_sort_col='st_level0_key',
        
        color_by_model=True
    )
#%%
print("Creating Entity Radar Chart...")
create_and_save_radar_chart_with_ring(
        df, 
        coarse_color_map=ENTITY_COARSE_COLOR_MAP, 
        save_path='entity_l2_radar_chart_20251109_2101.png', 
        
        # Columns for labels and colors
        category_col='entity_level1_cls',
        coarse_category_col='entity_level0_cls', 
        
        # NEW: Columns for sorting
        category_sort_col='entity_level1_key',
        coarse_sort_col='entity_level0_key',
        
        color_by_model=True
    )

# %%


# %%
import matplotlib.patheffects as pe
# CONFIGURATION FOR ST CHART
FIGSIZE = (12, 14)
UNIVERSAL_FONTSIZE = 30
LEGEND_POS=(1.17, 0.9)
LABEL_RADIUS = 1.42
TEXT_RADIUS = 0.8
OUTER_RADIUS = 1.05

def create_and_save_radar_chart_with_ring(df, 
                                          coarse_color_map, 
                                          save_path=None, 
                                          category_col='fine_category', 
                                          coarse_category_col='coarse_category',
                                          category_sort_col=None,
                                          coarse_sort_col=None,
                                          color_by_model=True,
                                          universal_fontsize=UNIVERSAL_FONTSIZE,
                                          debug=False):
    """
    Generates a radar chart with an outer ring, sorted by the
    provided key columns using natural sort.
    """
    
    if debug:
        print(f"--- DEBUG: Running for {save_path} ---")
        print(f"Coarse Col (Label): {coarse_category_col}")
        print(f"Fine Col (Label):   {category_col}")
        print(f"Coarse Col (Sort):  {coarse_sort_col}")
        print(f"Fine Col (Sort):    {category_sort_col}")
        
    sort_col_fine = category_sort_col if category_sort_col else category_col
    sort_col_coarse = coarse_sort_col if coarse_sort_col else coarse_category_col
    
    required_cols = [category_col, coarse_category_col, sort_col_fine, sort_col_coarse]
    
    for col in set(required_cols):
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in DataFrame.")
            return

    plot_df = df.dropna(subset=list(set(required_cols)))
    if plot_df.empty: 
        print(f"Skipping chart for '{category_col}': No data after dropping NaN."); 
        return

    # --- START OF SORTING LOGIC ---
    
    cols_to_get = [coarse_category_col, category_col, sort_col_coarse, sort_col_fine]
    category_df = plot_df[cols_to_get].drop_duplicates()

    def natural_sort_key(key_str):
        try:
            parts = tuple(int(part) for part in str(key_str).split('.'))
            return parts
        except (ValueError, TypeError):
            return (str(key_str),)

    sort_key_coarse_temp = '__sort_key_coarse'
    sort_key_fine_temp = '__sort_key_fine'
    
    category_df[sort_key_coarse_temp] = category_df[sort_col_coarse].apply(natural_sort_key)
    category_df[sort_key_fine_temp] = category_df[sort_col_fine].apply(natural_sort_key)
    
    category_df = category_df.sort_values(by=[sort_key_coarse_temp, sort_key_fine_temp])
    
    categories = category_df[category_col].tolist()
    
    if debug:
        print(f"\n[DEBUG] Final sorted category list ({len(categories)} total):")
        print(categories)
        print("--- END DEBUG ---")
    
    # --- END OF SORTING LOGIC ---

    models = plot_df['model'].unique(); models.sort()
    
    if len(categories) < 3: 
        print(f"Skipping chart for '{category_col}': Not enough categories."); 
        return
    
    pivot_df = plot_df.pivot_table(index='model', columns=category_col, values='mvIoU', aggfunc='mean')
    pivot_df = pivot_df.reindex(columns=categories, fill_value=0) 
    
    sorted_coarse_categories = category_df[coarse_category_col].tolist()
    
    if len(categories) != len(sorted_coarse_categories):
        print("Error: Category list length mismatch. Aborting.")
        return

    max_iou = pivot_df.values.max(); data_max = max_iou + 0.05
    if data_max == 0.05: data_max = 1.0

    num_vars = len(categories)
    
    angles_plot = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist(); angles_plot += angles_plot[:1] 
    angles_labels = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    
    figsize = FIGSIZE
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for i, model in enumerate(models):
        values = pivot_df.loc[model].tolist(); values += values[:1]
        
        color_map = plt.colormaps.get('tab20'); 
        color = color_map(i % color_map.N) if color_by_model else 'blue'
        
        ax.plot(angles_plot, values, color=color, linewidth=2, label=model, zorder=2)
        ax.fill(angles_plot, values, color=color, alpha=0.25, zorder=2)

    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_rlabel_position(0)
    
    ax.set_xticks(angles_labels)
    ax.set_xticklabels([]) 

    arc_inner_radius = data_max
    arc_thickness_ratio = 0.08 
    arc_outer_radius = data_max * (1 + arc_thickness_ratio)
    
    segment_width = 2 * np.pi / num_vars
    start_angles = angles_labels

    for i in range(num_vars):
        # This now gets the correct coarse label for each slice
        coarse_cat = sorted_coarse_categories[i] 
        color = coarse_color_map.get(coarse_cat, '#808080') 
        ax.bar(
            x=start_angles[i],
            height=arc_outer_radius - arc_inner_radius,
            width=segment_width,
            bottom=arc_inner_radius,
            color=color,
            alpha=0.7,
            zorder=1,
            align='edge'
        )


    ax.spines['polar'].set_visible(False)

    label_radius = arc_inner_radius * LABEL_RADIUS
    
    for i, category in enumerate(categories):
        angle_rad = angles_labels[i] 
        ax.text(angle_rad, 
                label_radius,
                category,
                fontsize=universal_fontsize,
                fontweight='light',
                rotation=0,
                ha='center',   
                va='center',  
                color='black',
                zorder=3)
    coarse_blocks = []
    if sorted_coarse_categories:
        current_block_label = sorted_coarse_categories[0]
        current_block_start_index = 0
        for i in range(1, num_vars):
            if sorted_coarse_categories[i] != current_block_label:
                coarse_blocks.append({
                    'label': current_block_label,
                    'start': current_block_start_index,
                    'end': i - 1
                })
                current_block_label = sorted_coarse_categories[i]
                current_block_start_index = i
        # Add the last block
        coarse_blocks.append({
            'label': current_block_label,
            'start': current_block_start_index,
            'end': num_vars - 1
        })

        # Handle wrap-around (if 'Spatial' is at the end and beginning)
        first_block = coarse_blocks[0]
        last_block = coarse_blocks[-1]
        if len(coarse_blocks) > 1 and first_block['label'] == last_block['label']:
            first_block['start'] = last_block['start']
            coarse_blocks.pop() # Remove the (now merged) last block

    # 2. Plot the label for each block
    text_radius = arc_outer_radius * TEXT_RADIUS # Place text just outside the ring

    for block in coarse_blocks:
        label = block['label']
        color = coarse_color_map.get(label, '#808080')
        start_index = block['start']
        end_index = block['end']

        # Find the middle angle of the block
        angle_start = angles_labels[start_index]
        angle_end = angles_labels[end_index] + segment_width # Add width for 'edge' align

        # Handle wrap-around case for middle angle calculation
        if angle_end < angle_start: # This block wraps around 2pi
            middle_angle = (angle_start + angle_end + 2 * np.pi) / 2
        else:
            middle_angle = (angle_start + angle_end) / 2
        
        middle_angle = middle_angle % (2 * np.pi) # Normalize angle

        ax.text(middle_angle, 
                text_radius,
                label,
                fontsize=universal_fontsize,
                fontweight='bold', # Make it bold to stand out
                rotation=0,
                ha='center',   
                va='center',  
                color=color, # Use the ring color
                zorder=3,
                path_effects=[pe.withStroke(linewidth=3, foreground='black')])
    
    ax.set_yticks(np.arange(0, data_max, data_max / 5))
    ax.set_yticklabels([f'{val:.2f}' for val in np.arange(0, data_max, data_max / 5)], color="grey", size=universal_fontsize)
    
    final_chart_max = arc_outer_radius * OUTER_RADIUS
    ax.set_ylim(0, final_chart_max)
    
    ax.legend(
        loc='center left', 
        bbox_to_anchor=LEGEND_POS, 
        fontsize=universal_fontsize,
        frameon=False # Border removed as requested
    )
    # fig.tight_layout(rect=[0, 0, 0.8, 1])

    if save_path:
        fig.savefig(save_path, format='png', bbox_inches='tight'); plt.close(fig)
        print(f"Radar chart saved to {save_path}")
    else: 
        plt.show()

print("Creating Entity Radar Chart...")


# %%
import matplotlib.patheffects as pe
# CONFIGURATION FOR ST CHART
FIGSIZE = (12, 14)
UNIVERSAL_FONTSIZE = 25
LEGEND_POS=(1.1, 0.8)
LABEL_RADIUS = 1.1
TEXT_RADIUS = 0.8
OUTER_RADIUS = 1.01

create_and_save_radar_chart_with_ring(
        df, 
        coarse_color_map=ST_COARSE_COLOR_MAP, 
        save_path='st_l2_radar_chart_20251109_1017.png', 

        
        # Columns for labels and colors
        category_col='st_level1_cls',
        coarse_category_col='st_level0_cls', 
        
        # NEW: Columns for sorting
        category_sort_col='st_level1_key',
        coarse_sort_col='st_level0_key',
        
        color_by_model=True,
        debug=True
        
    )


# %%
import matplotlib.patheffects as pe
# CONFIGURATION FOR ENTITY CHART
FIGSIZE = (12, 14)
UNIVERSAL_FONTSIZE = 30
LEGEND_POS=(1.17, 0.9)
LABEL_RADIUS = 1.42
TEXT_RADIUS = 0.8
OUTER_RADIUS = 1.05

create_and_save_radar_chart_with_ring(
        df, 
        coarse_color_map=ENTITY_COARSE_COLOR_MAP, 
        save_path='entity_l2_radar_chart_20251109_1013.png', 

        
        # Columns for labels and colors
        category_col='entity_level1_cls',
        coarse_category_col='entity_level0_cls', 
        
        # NEW: Columns for sorting
        category_sort_col='entity_level1_key',
        coarse_sort_col='entity_level0_key',
        
        color_by_model=True,
        debug=True
        
    )


# %%
