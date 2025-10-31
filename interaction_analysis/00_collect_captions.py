"""
This file gathers all the captions across HC-STVG-1, HC-STVG-2, VidSTG, VidVRD, and Something Else.
Each caption is saved as an object containing the caption text and its source dataset name.
"""

import json
import os
from tqdm import tqdm

# --------------------------
# Dataset Annotation Paths
# --------------------------
# Paths for the "freeform" task annotations for each dataset
DATASET_PATHS = {
    "hcstvg1": "/home/c3-0/datasets/stvg/hcstvg1/test_proc.json",
    "hcstvg2": "/home/c3-0/datasets/stvg/hcstvg2/annotations/HCVG_val_proc.json",
    "vidvrd": "/home/we337236/stvg/dataset/vidvrd/free_form_final_output.json",
    # "sthelse": "/home/c3-0/datasets/stvg/something_else/validation_proc.json",
    "vidstg": "/home/we337236/stvg/dataset/vidstg/vidstg_pro_test_final_list.json",
}

# Output file where all captions will be stored
OUTPUT_FILE = "hcstvg1_hcstvg2_vidvrd_vidstg_captions_with_dataset.json"

def gather_all_captions(paths, output_filename):
    """
    Opens multiple dataset annotation files, extracts all captions,
    and saves them to a single JSON file as objects containing the
    caption and its source dataset.

    Args:
        paths (dict): A dictionary where keys are dataset names and
                      values are file paths to the JSON annotations.
        output_filename (str): The name of the file to save the combined captions.
    """
    all_captions = []
    
    print("Starting to gather captions from all datasets...")

    # Iterate over each dataset path in the dictionary
    for dataset_name, file_path in paths.items():
        print(f"\nProcessing dataset: {dataset_name}...")
        
        # Check if the file exists before trying to open it
        if not os.path.exists(file_path):
            print(f"Warning: File not found for '{dataset_name}' at path: {file_path}. Skipping.")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Use tqdm for a progress bar while iterating through entries
                for entry in tqdm(data, desc=f"Extracting from {dataset_name}"):
                    caption_text = None
                    # Check if the 'caption' key exists and is not empty
                    if "caption" in entry and entry["caption"]:
                        caption_text = entry["caption"].strip()
                    # Handle cases where the key might be 'phrases' instead
                    elif "phrases" in entry and entry["phrases"]:
                       # Assuming the first phrase is the relevant one
                       caption_text = entry["phrases"][0].strip()

                    # If a valid caption was found, append it as an object
                    if caption_text:
                        all_captions.append({
                            "caption": caption_text,
                            "dataset": dataset_name
                        })

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from file: {file_path}. It might be corrupted.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

    # Write all the collected caption objects to the output file
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_captions, f, indent=4)
        print(f"\nSuccessfully gathered {len(all_captions)} captions.")
        print(f"All captions have been saved to '{output_filename}'.")
    except Exception as e:
        print(f"An error occurred while writing to the output file: {e}")


if __name__ == "__main__":
    gather_all_captions(DATASET_PATHS, OUTPUT_FILE)