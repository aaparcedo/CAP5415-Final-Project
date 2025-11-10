# %%
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
import seaborn as sns
from tqdm import tqdm
 # %%
# ---
# ---  SINGLE CONTROL PANEL  ---
# ---
CONTROL_PANEL = {
    # --- Data Directories (from your script) ---
    "dalton_results_dir": "/home/aparcedo/IASEB/results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/results/postprocessed/final_aka_on_paper/detection",

    # --- Area Ratio Files (from our previous script) ---
    # Put the full path to these files
    "area_ratio_files": {
        "hcstvg1": "hcstvg1_bbox_area_ratio_test_proc.json",
        "hcstvg2": "hcstvg2_bbox_area_ratio_HCVG_val_proc.json",
        "vidvrd": "vidvrd_bbox_area_ratio_free_form_final_output.json",
        "vidstg": "vidstg_bbox_area_ratio_pro_test_final_list.json"
    },
    
    # --- Output Filename ---
    "output_plot_filename": "performance_vs_bbox_ratio.png"
}

# ---
# --- END CONTROL PANEL ---
# ---


def normalize_caption(caption):
    """Helper function to exactly match your normalization logic."""
    if not isinstance(caption, str):
        return ""
    return caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '')


def load_area_ratio_data(area_ratio_files_map):
    """
    Loads all area ratios into a dictionary of lists, one per dataset.
    Returns: {'hcstvg1': [0.1, 0.2, ...], 'hcstvg2': [0.3, ...], ...}
    """
    ratio_lists_by_dataset = {}
    print("Loading area ratio data...")
    
    for dataset_name, filepath in area_ratio_files_map.items():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f) # data is a list
        except FileNotFoundError:
            print(f"ERROR: Area ratio file not found. Halting: {filepath}")
            raise
        
        print(f"  Processing ratios from {dataset_name} ({filepath})")
        
        # Create the list of ratios for this dataset
        ratios = []
        for i, entry in enumerate(data):
            ratio = entry.get("avg_bbox_area_ratio")
            # Strict check as requested: fail if any ratio is missing or invalid
            assert ratio is not None and not np.isnan(ratio), \
                f"Data integrity error in {filepath}: Entry {i} is missing a valid ratio."
            
            # We don't check for caption here, as we no longer use it for matching
            ratios.append(ratio)
            
        print(f"    Found {len(ratios)} ratios.")
        ratio_lists_by_dataset[dataset_name] = ratios
        
    print("Area ratio loading complete.")
    return ratio_lists_by_dataset


def load_performance_data(dalton_dir, alejandro_dir, ratio_lists_by_dataset):
    """
    Loads performance results and matches them with ratios by index.
    Returns a DataFrame: [model, dataset, caption, mvIoU, ratio]
    """
    processed_records = []

    # --- READ DALTON RESULTS ---
    print("Processing Dalton results...")
    try:
        filenames = os.listdir(dalton_dir)
        ff_filenames = [fn for fn in filenames if "freeform" in fn]
    except FileNotFoundError:
        print(f"Warning: Dalton results directory not found: {dalton_dir}")
        ff_filenames = []

    for ff_fn in ff_filenames:
        dataset = ff_fn.split("_")[1]
        print(f'Loading dataset: {dataset}')
        if dataset not in ratio_lists_by_dataset:
            print(f"Warning: No ratios found for dataset '{dataset}'. Skipping file: {ff_fn}")
            continue

        ratios = ratio_lists_by_dataset[dataset]
        filepath = os.path.join(dalton_dir, ff_fn)
        with open(filepath, 'r') as f:
            data = json.load(f) # data is a list

        # Check for parallel list integrity
        assert len(data) == len(ratios), \
            f"Mismatched length error: {ff_fn} has {len(data)} entries, but its ratio file has {len(ratios)}."

        for i, sample_dict in enumerate(data):
            processed_records.append({
                "model": ff_fn.split("_")[0],
                "dataset": dataset,
                "caption": normalize_caption(sample_dict.get("caption")),
                "mvIoU": sample_dict.get("mvIoU"),
                "ratio": ratios[i] # Match by index
            })

    # --- READ ALEJANDRO RESULTS ---
    print("Processing Alejandro results...")
    try:
        filenames = os.listdir(alejandro_dir)
        ff_filenames = [fn for fn in filenames if "freeform" in fn]
    except FileNotFoundError:
        print(f"Warning: Alejandro results directory not found: {alejandro_dir}")
        ff_filenames = []

    for ff_fn in ff_filenames:
        dataset = ff_fn.split("_")[2]
        if dataset not in ratio_lists_by_dataset:
            print(f"Warning: No ratios found for dataset '{dataset}'. Skipping file: {ff_fn}")
            continue

        ratios = ratio_lists_by_dataset[dataset]
        filepath = os.path.join(alejandro_dir, ff_fn)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "results" not in data:
            print(f"Warning: 'results' key not in {ff_fn}. Skipping.")
            continue
        
        results_list = data["results"]

        # Check for parallel list integrity
        assert len(results_list) == len(ratios), \
            f"Mismatched length error: {ff_fn} has {len(results_list)} entries, but its ratio file has {len(ratios)}."

        for i, sample_dict in enumerate(results_list):
            processed_records.append({
                "model": ff_fn.split("_")[1],
                "dataset": dataset,
                "caption": normalize_caption(sample_dict.get("caption")),
                "mvIoU": sample_dict.get("mvIoU", sample_dict.get("mvIoU_tube_step")),
                "ratio": ratios[i] # Match by index
            })

    df = pd.DataFrame(processed_records)
    print(f'Performance data loaded and matched. Found {len(df)} records.')
    return df


# ---
# --- Main Execution ---
# ---
# %%
# 1. Load area ratio data FIRST
ratio_lists = load_area_ratio_data(
    CONTROL_PANEL["area_ratio_files"]
)

# 2. Load performance data and pass ratios to it
perf_df = load_performance_data(
    CONTROL_PANEL["dalton_results_dir"],
    CONTROL_PANEL["alejandro_results_dir"],
    ratio_lists
)
# %%
if perf_df.empty:
    print("Error: No data loaded. Cannot create plot. Check file paths and assertions.")
else:
    # 3. Clean the combined data
    orig_len = len(perf_df)
    merged_df = perf_df.dropna(subset=['ratio', 'mvIoU'])
    print(f"Cleaning complete. Using {len(merged_df)} out of {orig_len} records (dropped rows with null mvIoU or ratio).")

    if merged_df.empty:
        print("Error: No valid data after dropping nulls.")
    else:
        # 4. Bin the Data
        # --- FIX 1: Use 21 for 20 bins ---
        bins = np.linspace(0.0, 1.0, 21) # 21 edges = 20 bins
        
        # Using .2f in labels to show 0.05, 0.10, etc. (optional)
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
        
        print(f"Number of bins: {len(labels)}")
        
        merged_df["ratio_bin"] = pd.cut(
            merged_df["ratio"],
            bins=bins,
            labels=labels,
            right=True
        )

        # 5. Plot the Data
        print("Creating performance plot...")
        
        # Set Seaborn theme
        sns.set_theme(style="darkgrid", rc={
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linestyle': '-',
        })
        plt.figure(figsize=(12, 7))

        plot_color = "#8da2cc"

        ax = sns.barplot(
            data=merged_df,
            x="ratio_bin",
            y="mvIoU",
            color=plot_color,
            errorbar=None,
            width=0.9
        )

        # --- FIX 2: Update tick positions for 20 bins ---
        # We map the labels to the correct bar indices
        # (0, 4, 8, 12, 16) and add the 1.0 label at the end (19.5)
        tick_positions = [0, 4, 8, 12, 16]
        tick_labels = ["0.0", "0.2", "0.4", "0.6", "0.8"]
        
        # Add the 1.0 label at the very end of the axis
        # The axis for 20 bars (0-19) ends at 19.5
        tick_positions.append(19.5)
        tick_labels.append("1.0")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        # We also clear the minor ticks that might show up
        ax.set_xticks([], minor=True)
        # --- END FIX ---
        
        # 3. Set titles and labels on the 'ax' object
        ax.set_title("Model Performance (mvIoU) vs. BBox Area Ratio\ngdino, llava, shikra, cogvlm, ferret\ndetection only: vg1&2, vrd,stg")
        ax.set_xlabel("Average BBox Area Ratio (Binned)", fontsize=14)
        ax.set_ylabel("Average mvIoU", fontsize=14)
        
        plt.tight_layout()

        # 6. Save the figure
        output_filename = CONTROL_PANEL["output_plot_filename"]
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"\nSuccess! Plot saved to: {output_filename}")
        # plt.show()
# %%
