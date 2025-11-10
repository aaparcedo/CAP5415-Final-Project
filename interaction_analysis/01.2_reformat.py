# %%
from IASEB.datasets import DATASET_PATHS
import json
import os

datapath = "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_captions.json"
captions_data = json.load(open(datapath, 'r'))
# %%

reformatted_samples = []

stvg1_data = json.load(open(DATASET_PATHS['hcstvg1']['freeform']))
stvg2_data = json.load(open(DATASET_PATHS['hcstvg2']['freeform']))

vrd_data = json.load(open(DATASET_PATHS['vidvrd']['freeform']))

stg_data = json.load(open(DATASET_PATHS['vidstg']['freeform']))

mevis_data = json.load(open(DATASET_PATHS['mevis']['bbox']))
rvos_data = json.load(open(DATASET_PATHS['rvos']['bbox']))

det_datasets = stvg1_data + stvg2_data + vrd_data + stg_data 

index = 0

for sample in det_datasets:
    if sample["caption"] != captions_data[index]["caption"]: # only doing this to make sure that i load everything in the same way i did it originally
        print(sample)
        print(captions_data[index])
        print(index)
        break
    
    caption_raw = sample["caption"]
    caption = caption_raw.strip().lower().replace('.', '').replace('"', '').replace('\\', '')

    reformat_sample = sample.copy()
    

    if 'st_frame' in sample:
        reformat_sample["tube_start_frame"] = sample['st_frame']
        reformat_sample["tube_end_frame"] = sample['ed_frame']
        reformat_sample.pop('st_frame')
        reformat_sample.pop('ed_frame')

    if "trajectory" in reformat_sample:
        reformat_sample["gt_tube"] = {frame_id: gt_bbox for frame_id, gt_bbox in zip(
                                            range(reformat_sample["tube_start_frame"], reformat_sample["tube_end_frame"]+1),
                                            reformat_sample["trajectory"])}
        reformat_sample.pop("trajectory")
        
    elif "bbox" in reformat_sample:
        reformat_sample["gt_tube"] = reformat_sample["bbox"]
        reformat_sample.pop("bbox")
    else:
        print('no gt tube found in sample')
    
    if sample in stvg1_data: 
        reformat_sample['dataset'] = 'hcstvg1'
        reformat_sample["video_path"] = os.path.join(DATASET_PATHS['hcstvg1']['video'], reformat_sample["video_path"])
    elif sample in stvg2_data: 
        reformat_sample['dataset'] = 'hcstvg2'
        reformat_sample["video_path"] = os.path.join(DATASET_PATHS['hcstvg2']['video'], reformat_sample["video_path"])
    elif sample in vrd_data: 
        reformat_sample['dataset'] = 'vidvrd'
    elif sample in stg_data: 
        reformat_sample['dataset'] = 'vidstg'
    reformatted_samples.append(reformat_sample)
    index += 1
# %%
dataset = []
# mevis
mevis_data_fp = "/share/datasets/stvg/mevis_annotations/valid_u/one_obj_bbox_updated_format.json"
mevis_data = json.load(open(mevis_data_fp, 'r'))

for video_id, video_data in mevis_data["videos"].items():
    for exp_id, exp_data in video_data["expressions"].items():
        video_path = os.path.join(DATASET_PATHS['mevis']['video'], video_id)
        frame_ids = sorted(list(exp_data['trajectory'].keys()))
        sample = {
            "dataset": 'mevis',
            'video_id': video_id,
            'video_path': video_path,
            'caption': exp_data['exp'],
            'exp_id': exp_id,
            'obj_id': exp_data['obj_id'],
            'anno_id': exp_data['anno_id'],
            'tube_start_frame': frame_ids[0],
            'tube_end_frame': frame_ids[-1],
            'gt_tube': exp_data['trajectory']
        }


        reformatted_samples.append(sample)

rvos_data_fp = "/share/datasets/stvg/rvos_annotations/valid/rvos_bbox_annotations.json"
rvos_data = json.load(open(rvos_data_fp, 'r'))

# rvos
for video_id, video_data in rvos_data["videos"].items():
    for exp_id, exp_data in video_data["expressions"].items():
        video_path = os.path.join(DATASET_PATHS['rvos']['video'], video_id)
        frame_ids = sorted(list(exp_data['trajectory'].keys()))
        sample = {
            "dataset": 'rvos',
            'video_id': video_id,
            'video_path': video_path,
            'caption': exp_data['exp'],
            'exp_id': exp_id,
            'obj_id': exp_data['obj_id'],
            'tube_start_frame': frame_ids[0],
            'tube_end_frame': frame_ids[-1],
            'gt_tube': exp_data['trajectory']
        }

print(f'len of reformatted samples: {len(reformatted_samples)}')

output_path = "vg12_vrd_stg_mevis_rvos_metadata_1.json"


# %%%
with open(output_path, 'w') as f:
    json.dump(reformatted_samples, f, indent=4)
# %%
