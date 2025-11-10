# %%
import json

dataset = []
# mevis
mevis_data_fp = "/share/datasets/stvg/mevis_annotations/valid_u/one_obj_bbox_updated_format.json"
mevis_data = json.load(open(mevis_data_fp, 'r'))

for video_id, video_data in mevis_data["videos"].items():
    for exp_id, exp_data in video_data["expressions"].items():

        trajectory_dict = exp_data['trajectory']
        # Extract frame numbers from trajectory keys
        frame_numbers = sorted([int(k) for k in trajectory_dict.keys()])
        start_frame = frame_numbers[0]
        end_frame = frame_numbers[-1]
        
        # Calculate count and duration based on your provided formulas
        frame_count = end_frame - start_frame

        sample = {
            "dataset": 'mevis',
            'video_id': video_id,
            'caption': exp_data['exp'],
            "frame_count": frame_count,
            "start_frame": start_frame,
            "end_frame": end_frame,
            'exp_id': exp_id,
            'obj_id': exp_data['obj_id'],
            'anno_id': exp_data['anno_id'],
            # 'trajectory': exp_data['trajectory']
        }
        dataset.append(sample)

rvos_data_fp = "/share/datasets/stvg/rvos_annotations/valid/rvos_bbox_annotations.json"
rvos_data = json.load(open(rvos_data_fp, 'r'))
# rvos
for video_id, video_data in rvos_data["videos"].items():
    for exp_id, exp_data in video_data["expressions"].items():

        trajectory_dict = exp_data['trajectory']
        # Extract frame numbers from trajectory keys
        frame_numbers = sorted([int(k) for k in trajectory_dict.keys()])
        start_frame = frame_numbers[0]
        end_frame = frame_numbers[-1]
        
        # Calculate count and duration based on your provided formulas
        frame_count = end_frame - start_frame
        sample = {
            "dataset": 'rvos',
            'video_id': video_id,
            'caption': exp_data['exp'],
            "frame_count": frame_count,
            "start_frame": start_frame,
            "end_frame": end_frame,
            'exp_id': exp_id,
            'obj_id': exp_data['obj_id'], 
            # 'trajectory': exp_data['trajectory']
        }
        dataset.append(sample)


output_path = "mevis_rvos_captions.json"

with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=4)
# %%
