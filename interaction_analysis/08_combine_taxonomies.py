# This is a one time use script to 
# (1) include all available sample metadata in main json that will be used for analysis
# (2) combine st and entity taxonomies into a single file
# (3) combine det datasets and seg datasets classifications into the same file

# %%
# Aggregate all the detection and results for IASEB 
# Two main sections: detection & segmentation
import json
import pandas as pd
import os
from tqdm import tqdm 
from pathlib import Path 
from constants import ST_HIERARCHY, ENTITY_HIERARCHY, DATASET_PATHS



# THIS CAN BE SIMPLIFIED TO USE /home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_taxonomies_v1.json
# CAN ALSO BE USED TO constants.py
CONTROL_PANEL = {
    "st_data_path": "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_st_class_v1.json",
    "entity_data_path": "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_v1.json",

    # --- Data Directories ---
    "dalton_results_dir": "/home/aparcedo/IASEB/results/all_final_results/llava_gdino_dalton_interpolated_results",
    "alejandro_results_dir": "/home/aparcedo/IASEB/results/all_final_results/final_aka_on_paper_alejandro/detection",
    "wen_results_dir": "/home/aparcedo/IASEB/results/all_final_results/stvg_output_bbox_wen",
    "anirudh_results_dir": "/home/aparcedo/IASEB/results/all_final_results/STVG_results_anirudh"
}

# %%
# --- 1. LOAD AND VALIDATE CLASSIFICATION DATA ---
# THE VALIDATION PART OF THIS CELL SHOULD BE ITS OWN STEP IN THE CLASSIFICATION PROCESS
# WE SHOULDNT BE CHECKING IF THE CLASSIFICATION WAS VALID THIS LATE

st_hierarchy = ST_HIERARCHY
entity_hierarchy = ENTITY_HIERARCHY

def is_valid_path(hierarchy, path_str):
    """
    Checks if a category path string (e.g., "1.2.1") is valid
    by traversing the given hierarchy.
    """
    if not path_str:
        return False
        
    # Handle paths that might have notes, e.g., "1.2.1 Note..."
    path_str_clean = path_str.split(" ")[0] 
    
    try:
        current_level_dict = hierarchy
        levels = path_str_clean.split('.')
        for level_key_str in levels:
            key = int(level_key_str)
            node = current_level_dict[key]
            current_level_dict = node['children']
        return True
    except (KeyError, ValueError, TypeError, AttributeError):
        return False

# Load ST Classification Map (Raw)
st_classification_data = json.load(open(CONTROL_PANEL["st_data_path"], 'r'))

# need a dict of caption : category
raw_st_cls_data_map = {}
for entry in st_classification_data:
    caption = entry["caption"]
    key = caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
    raw_st_cls_data_map[key] = entry.get("category", entry.get("st_class_raw"))
    
# Load Entity Classification Map (Raw)
entity_classification_data = json.load(open(CONTROL_PANEL["entity_data_path"], 'r'))
raw_entity_cls_data_map = {}
for entry in entity_classification_data:
    caption = entry["caption"]
    key = caption.strip().lower().replace('.', '').replace('"', '').replace('\\', '') 
    raw_entity_cls_data_map[key] = entry.get("category", entry.get("entity_class_raw"))

print(f"Original ST classifications: {len(raw_st_cls_data_map)}")
print(f"Original Entity classifications: {len(raw_entity_cls_data_map)}")

# --- Filter ST Map for valid paths ---
st_cls_data_map = {}
invalid_st_count = 0
for caption, category_str in raw_st_cls_data_map.items():
    if is_valid_path(st_hierarchy, category_str):
        st_cls_data_map[caption] = category_str
    else:
        print(f"Invalid ST Path: {category_str} for caption: {caption}")
        invalid_st_count += 1

# --- Filter Entity Map for valid paths ---
entity_cls_data_map = {}
invalid_entity_count = 0
for caption, category_str in raw_entity_cls_data_map.items():
    if is_valid_path(entity_hierarchy, category_str):
        entity_cls_data_map[caption] = category_str
    else:
        # print(f"Invalid Entity Path: {category_str} for caption: {caption}")
        invalid_entity_count += 1

print("--- VALIDATION COMPLETE ---")
print(f"Invalid ST classifications removed: {invalid_st_count}")
print(f"Invalid Entity classifications removed: {invalid_entity_count}")
print(f"Clean ST classifications remaining: {len(st_cls_data_map)}")
print(f"Clean Entity classifications remaining: {len(entity_cls_data_map)}")
print("---------------------------")



# %%
# MOVING ALL OF THIS TO 01.2 except the part where we filter by class. that should stay somewhere around here. 
# datapath = "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_captions.json"
# captions_data = json.load(open(datapath, 'r'))
# # %%

# reformatted_samples = []

# stvg1_data = json.load(open(DATASET_PATHS['hcstvg1']['freeform']))
# stvg2_data = json.load(open(DATASET_PATHS['hcstvg2']['freeform']))

# vrd_data = json.load(open(DATASET_PATHS['vidvrd']['freeform']))

# stg_data = json.load(open(DATASET_PATHS['vidstg']['freeform']))

# mevis_data = json.load(open(DATASET_PATHS['mevis']['bbox']))
# rvos_data = json.load(open(DATASET_PATHS['rvos']['bbox']))

# det_datasets = stvg1_data + stvg2_data + vrd_data + stg_data 

# index = 0

# for sample in det_datasets:
#     if sample["caption"] != captions_data[index]["caption"]: # only doing this to make sure that i load everything in the same way i did it originally
#         print(sample)
#         print(captions_data[index])
#         print(index)
#         break
#     index += 1
#     caption_raw = sample["caption"]
#     caption = caption_raw.strip().lower().replace('.', '').replace('"', '').replace('\\', '')

#     reformat_sample = sample.copy()
#     reformat_sample["frame_count"] = captions_data[index]["frame_count"]
#     reformat_sample["start_frae"] = captions_data[index]["start_frame"]
#     reformat_sample["end_frame"] = captions_data[index]["end_frame"]
#     reformat_sample.pop("trajectory", None) # REMOVE THESE IF I WANT TO SAVE THE BOXES
#     reformat_sample.pop("bbox", None)
#     if sample in stvg1_data: 
#         reformat_sample['dataset'] = 'hcstvg1'
#     elif sample in stvg2_data: 
#         reformat_sample['dataset'] = 'hcstvg2'
#     elif sample in vrd_data: 
#         reformat_sample['dataset'] = 'vidvrd'
#     elif sample in stg_data: 
#         reformat_sample['dataset'] = 'vidstg'
#     reformatted_samples.append(reformat_sample)


# import json

# dataset = []
# # mevis
# mevis_data_fp = "/share/datasets/stvg/mevis_annotations/valid_u/one_obj_bbox_updated_format.json"
# mevis_data = json.load(open(mevis_data_fp, 'r'))
# mevis_count = 0
# mevis_total = 0
# for video_id, video_data in mevis_data["videos"].items():
#     for exp_id, exp_data in video_data["expressions"].items():
#         mevis_total += 1
#         video_path = os.path.join(DATASET_PATHS['mevis']['video'], video_id)
#         sample = {
#             "dataset": 'mevis',
#             'video_id': video_id,
#             'video_path': video_path,
#             'caption': exp_data['exp'],
#             'exp_id': exp_id,
#             'obj_id': exp_data['obj_id'],
#             'anno_id': exp_data['anno_id']
#         }

#         # this is our category look up key
#         caption_raw = exp_data['exp']
#         caption = caption_raw.strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        
#         # hash map look up
#         st_category_str = st_cls_data_map.get(caption)
#         entity_category_str = entity_cls_data_map.get(caption)

#         if not st_category_str:
#             # print(f"Skipping caption ST (no valid classification): {caption}")
#             continue
#         if not entity_category_str:
#             # print(f"Skipping caption ST (no valid classification): {caption}")
#             continue
#         # 
        
#         # 
#         st_levels = st_category_str.split(" ")[0].split('.')
#         try:
#             current_level_dict = st_hierarchy
#             for level_idx, category_num_str in enumerate(st_levels):
#                 key = int(category_num_str)
#                 node = current_level_dict[key]
#                 sample[f'st_cls_level{level_idx}'] = node['short_name']
#                 current_level_dict = node['children']
#         except (KeyError, ValueError, TypeError):
#             # This should no longer happen, but good to keep.
#             # print(f"Warning: SKIPPING invalid ST path '{st_category_str}' for caption '{caption}'.")
#             continue # Skip this record
                
        
#         # --- Add dynamic Entity levels by traversing the hierarchy ---
#         entity_levels = entity_category_str.split(" ")[0].split('.')
#         try:
#             current_level_dict = entity_hierarchy
#             for level_idx, category_num_str in enumerate(entity_levels):
#                 key = int(category_num_str)
#                 node = current_level_dict[key]
#                 sample[f'entity_cls_level{level_idx}'] = node['short_name']
#                 current_level_dict = node['children']
#         except (KeyError, ValueError, TypeError):
#             # This should no longer happen, but good to keep.
#             # print(f"Warning: SKIPPING invalid Entity path '{entity_category_str}' for caption '{caption}'.")
#             continue # Skip this record

#         reformatted_samples.append(sample)
#         mevis_count += 1

# print(f'mevis total: {mevis_total}')
# print(f'len of mevis samples saved: {mevis_count}')

# rvos_data_fp = "/share/datasets/stvg/rvos_annotations/valid/rvos_bbox_annotations.json"
# rvos_data = json.load(open(rvos_data_fp, 'r'))

# rvos_count = 0
# rvos_total = 0
# # rvos
# for video_id, video_data in rvos_data["videos"].items():
#     for exp_id, exp_data in video_data["expressions"].items():
#         rvos_total += 1
#         video_path = os.path.join(DATASET_PATHS['rvos']['video'], video_id)
#         sample = {
#             "dataset": 'rvos',
#             'video_id': video_id,
#             'video_path': video_path,
#             'caption': exp_data['exp'],
#             'exp_id': exp_id,
#             'obj_id': exp_data['obj_id']
#         }
#         # this is our category look up key
#         caption_raw = exp_data['exp']
#         caption = caption_raw.strip().lower().replace('.', '').replace('"', '').replace('\\', '')
        
#         # # hash map look up
#         # st_category_str = st_cls_data_map.get(caption)
#         # entity_category_str = entity_cls_data_map.get(caption)

#         # if not st_category_str:
#         #     # print(f"Skipping caption ST (no valid classification): {caption}")
#         #     continue
#         # if not entity_category_str:
#         #     # print(f"Skipping caption ST (no valid classification): {caption}")
#         #     continue
#         # # 
#         # st_levels = st_category_str.split(" ")[0].split('.')
#         # try:
#         #     current_level_dict = st_hierarchy
#         #     for level_idx, category_num_str in enumerate(st_levels):
#         #         key = int(category_num_str)
#         #         node = current_level_dict[key]
#         #         sample[f'st_cls_level{level_idx}'] = node['short_name']
#         #         current_level_dict = node['children']
#         # except (KeyError, ValueError, TypeError):
#         #     # This should no longer happen, but good to keep.
#         #     # print(f"Warning: SKIPPING invalid ST path '{st_category_str}' for caption '{caption}'.")
#         #     continue # Skip this record
                
        
#         # # --- Add dynamic Entity levels by traversing the hierarchy ---
#         # entity_levels = entity_category_str.split(" ")[0].split('.')
#         # try:
#         #     current_level_dict = entity_hierarchy
#         #     for level_idx, category_num_str in enumerate(entity_levels):
#         #         key = int(category_num_str)
#         #         node = current_level_dict[key]
#         #         sample[f'entity_cls_level{level_idx}'] = node['short_name']
#         #         current_level_dict = node['children']
#         # except (KeyError, ValueError, TypeError):
#         #     # This should no longer happen, but good to keep.
#         #     # print(f"Warning: SKIPPING invalid Entity path '{entity_category_str}' for caption '{caption}'.")
#         #     continue # Skip this record
#         # reformatted_samples.append(sample)
#         # rvos_count += 1

# print(f'rvos total samples: {rvos_total}')
# print(f'len rvos samples saved: {rvos_count}')

# print(f'len of reformatted samples: {len(reformatted_samples)}')

# output_path = "vg12_vrd_stg_mevis_rvos_gpt4omini_taxonomies_v1.json"

# # %%%
# with open(output_path, 'w') as f:
#     json.dump(reformatted_samples, f, indent=4)