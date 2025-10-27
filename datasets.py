
import json
import os
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, CLIPImageProcessor
import time
from utils import xywh_to_corners

# --------------------------
DEVICE = "cuda"
CAPTION_MOD_TEST1_PATH="/home/aparcedo/IASEB/test1_caption_modification_hcstvg1.json"

DATASET_PATHS = {
    "hcstvg1": {
        "video": "/home/c3-0/datasets/stvg/hcstvg1/v1/video",
        "referral": "/home/c3-0/datasets/stvg/preprocess_dump/hcstvg/hcstvg_pid_tubes_multi_sent_refined_v3/sentences_test.json", 
        "freeform": "/home/c3-0/datasets/stvg/hcstvg1/test_proc.json", 
    }, 
    "hcstvg2": {
        "video": "/home/c3-0/datasets/stvg/hcstvg2/videos",
        "referral": "/home/we337236/stvg/dataset/hcstvg_v2/hcstvgv2_sentences_test_gpt_modified.json", 
        "freeform": "/home/c3-0/datasets/stvg/hcstvg2/annotations/HCVG_val_proc.json", 
    }, 
    "vidstg": {
        "referral": "/share/datasets/stvg/vidstg_annotations/vidstg_referral.json", 
        "freeform": "/home/we337236/stvg/dataset/vidstg/vidstg_pro_test_final_list.json", 
    }, 
    "vidvrd": {
        "referral": "/home/we337236/stvg/dataset/vidvrd/referral_final_output.json", 
        "freeform": "/home/we337236/stvg/dataset/vidvrd/free_form_final_output.json", 
    }, 
    "mevis": {
        "video": "/share/datasets/stvg/MeViS/MeViS/valid_u/JPEGImages/JPEGImages",
        "metadata": "/share/datasets/stvg/mevis_annotations/valid_u/one_object_meta_expressions.json",
        "bbox": "/share/datasets/stvg/mevis_annotations/valid_u/one_obj_bbox_updated_format.json",
        "masks": "/share/datasets/stvg/mevis_annotations/valid_u/mask_dict.json"
    },
    "rvos": {
        "video": "/share/datasets/stvg/rvos_annotations/valid/JPEGImages",
        "masks": "/share/datasets/stvg/rvos_annotations/valid/Annotations",
        "metadata": "/share/datasets/stvg/rvos_annotations/valid/meta_expressions_challenge.json",
        "bbox": "/share/datasets/stvg/rvos_annotations/valid/rvos_bbox_annotations.json",
    }
}


class HCSTVGDataloader:
    def __init__(self, args):
        self.referral_data_path = DATASET_PATHS[args.dataset]["referral"]
        self.video_dir = DATASET_PATHS[args.dataset]["video"]
        self.frame_step = args.frame_step
        self.args = args
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        if self.args.task_type == 'referral':
            self.referral_caption_data = json.load(open(self.referral_data_path, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = os.path.join(self.video_dir, entry['video_path'])
        cap = cv2.VideoCapture(video_path)

        # we only change from the original caption in the referral task
        if self.args.task_type_type == 'referral':
            if self.args.dataset == 'hcstvg1':
                entry["caption"] = self.referral_caption_data[entry["original_video_id"]][0]["phrases"][0]
            elif self.args.dataset == 'hcstvg2':
                entry["caption"] = self.referral_caption_data[entry["original_video_id"]]
        
        bbox_data = entry.get("trajectory", [])
        start_frame = entry.get('tube_start_frame', 0)
        end_frame = entry.get('tube_end_frame', len(bbox_data) - 1 + start_frame)

        frames_with_gt = []
        sampled_gt_bboxs = []

        # Seek to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        
        # Manually track the frame index for reliability
        current_frame_idx = start_frame

        while cap.isOpened():
            # Stop if we have processed all frames in the annotated range
            if current_frame_idx > end_frame:
                break
            
            ret, frame = cap.read()

            if not ret:
                break
            
            if (current_frame_idx - start_frame) % self.frame_step == 0:
                # We use bounding box index here because bbox annotations are in an array
                # As opposed to a dict with {frame_id: bbox}
                box_index = current_frame_idx - start_frame
                if 0 <= box_index < len(bbox_data):
                    bbox = xywh_to_corners(bbox_data[box_index])
                    frames_with_gt.append((frame, bbox, current_frame_idx))
                    sampled_gt_bboxs.append(bbox)
            
            # Increment our manual counter
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs


# ------------------------------------------------------------------------
# VidVRD dataset class verified to work with free form and referral format.
# Have not yet verified that it works for phrase format (if any, not sure).
# Ground truth boxes are in (0..H, 0..W) range and [x, y, w, h] format.
# Converts from [x,y,w,h] to [xmin, ymin, xmax, ymax] during loading.
# ------------------------------------------------------------------------

class VidVRDDataloader:
    def __init__(self, args):
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        self.frame_step = args.frame_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]
        bbox_data = entry.get("bbox", {})
        start_frame = entry.get("st_frame", 0)
        end_frame = entry.get("ed_frame", 0)

        cap = cv2.VideoCapture(video_path)
        frames_with_gt = []
        sampled_gt_bboxs = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        current_frame_idx = start_frame

        while cap.isOpened():
            if current_frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame_idx - start_frame) % self.frame_step == 0:
                bbox_sample = bbox_data.get(str(current_frame_idx), [0, 0, 0, 0])
                bbox_sample = xywh_to_corners(bbox_sample)
                frames_with_gt.append((frame, bbox_sample, current_frame_idx))
                sampled_gt_bboxs.append(bbox_sample)
            
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs

# --------------------
# VidSTG Dataset Class
# --------------------

class VidSTGDataloader:
    def __init__(self, args):
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        self.frame_step = args.frame_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = entry["video_path"]
        bbox_data = entry["bbox"]
        start_frame = entry.get("st_frame", 0)
        end_frame = entry.get("ed_frame", 0)

        cap = cv2.VideoCapture(video_path)
        frames_with_gt = []
        sampled_gt_bboxs = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        current_frame_idx = start_frame

        while cap.isOpened():
            if current_frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame_idx - start_frame) % self.frame_step == 0:
                bbox_sample = bbox_data.get(str(current_frame_idx), [0, 0, 0, 0])
                converted_bbox = xywh_to_corners(bbox_sample)
                frames_with_gt.append((frame, converted_bbox, current_frame_idx))
                sampled_gt_bboxs.append(converted_bbox)
            
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs


class MeViSBBoxDataloader():
    """
    Only loads frame and GT bbox every frame_step.
    """
    def __init__(self, args):
        print("Loading MeViS annotations...")
        self.data = []
        self.args = args
        self.boxes = json.load(open(DATASET_PATHS[args.dataset]["bbox"], 'r'))
        self.metadata = json.load(open(DATASET_PATHS[args.dataset]["metadata"], 'r'))

        for video_id, video_md in self.metadata["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "video_id": video_id,
                    "video_path": os.path.join(DATASET_PATHS[args.dataset]["video"], video_id), 
                    "caption": exp_data["exp"],
                    "anno_id": exp_data["anno_id"][0],
                    "obj_id": exp_data["obj_id"][0],
                    "gt_bboxs": {frame_id: xywh_to_corners(bbox) for frame_id, bbox in self.boxes["videos"][video_id]["expressions"][exp_id]["trajectory"].items()},
                })

        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path =  entry["video_path"]
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sampled_gt_bboxs = {}

        all_frames_np = []
        for idx, (frame_id, gt_bbox) in enumerate(entry["gt_bboxs"].items()):
            if idx % self.args.frame_step == 0:
                frame = cv2.imread(os.path.join(video_path, image_files[int(frame_id)]))
                assert frame is not None, f'ERROR: {frame_id} for {video_path} is None'
                all_frames_np.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                sampled_gt_bboxs[frame_id] = gt_bbox

        entry["gt_bboxs"] = sampled_gt_bboxs

        return all_frames_np, entry


class ReferYouTubeVOSBBoxDataloader():
    def __init__(self, args):
        print("Loading Refer-YouTube-VOS metadata...")
        self.annotations = json.load(open(DATASET_PATHS[args.dataset]["bbox"], 'r'))
        self.data = []

        for video_id, video_md in self.annotations["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "exp_id": exp_id,
                    "video_id": video_id,
                    "video_path": os.path.join(DATASET_PATHS[args.dataset]["video"], video_id), 
                    "obj_id": exp_data["obj_id"],
                    "caption": exp_data["exp"],
                    "gt_bboxs": {frame_id: xywh_to_corners(bbox) for frame_id, bbox in exp_data["trajectory"].items()},

                })
        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        entry = self.data[idx]
        video_path =  entry["video_path"]
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sampled_gt_bboxs = {}

        # RVOS is presampled frame_step=5; we don't sample again
        all_frames_np = []
        for idx, (frame_id, gt_bbox) in enumerate(entry["gt_bboxs"].items()):
            try:
                frame = cv2.imread(os.path.join(video_path, image_files[idx]))
            except Exception as e:
                import code; code.interact(local=locals())
            assert frame is not None, f'ERROR: {frame_id} for {video_path} is None'
            all_frames_np.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sampled_gt_bboxs[frame_id] = gt_bbox

        entry["gt_bboxs"] = sampled_gt_bboxs

        return all_frames_np, entry

class STVGDataLoader:
    def __new__(cls, args):
        DATALOADER_MAP = {
            "hcstvg1": HCSTVGDataloader,
            "hcstvg2": HCSTVGDataloader,
            "vidstg": VidSTGDataloader,
            "vidvrd": VidVRDDataloader,
            "mevis": MeViSBBoxDataloader,
            "rvos": ReferYouTubeVOSBBoxDataloader
        }
        return DATALOADER_MAP[args.dataset](args)


class MeViSDataloader():
    def __init__(self, args):
        self.data = []
        self.args = args
        print("Loading MeViS annotations...")
        self.compressed_masks = json.load(open(DATASET_PATHS[self.args.dataset]["masks"], 'r'))
        self.metadata = json.load(open(DATASET_PATHS[self.args.dataset]["metadata"], 'r'))

        for video_id, video_md in self.metadata["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "video_id": video_id, 
                    "video_path": os.path.join(DATASET_PATHS[self.args.dataset]["video"], video_id), 
                    "caption": exp_data["exp"],
                    "anno_id": exp_data["anno_id"][0],
                    "obj_id": exp_data["obj_id"][0],
                })
        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = os.path.join(DATASET_PATHS[self.args.dataset]["video"], entry["video_id"])
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        anno_id_str = str(entry["anno_id"])
        assert anno_id_str in self.compressed_masks, f"ERROR: Annotation ID {anno_id_str} not in mask annotation JSON file."
        rle_masks = self.compressed_masks[anno_id_str]
        masks, frames = [], []
        
        for rle_mask, img_file in zip(rle_masks, image_files):
            # import code; code.interact(local=locals())
            if not rle_mask: continue # NOTE: mask might be empty (expected)
            masks.append(mask_util.decode(
                        {'size': rle_mask['size'], 'counts': rle_mask['counts'].encode('utf-8')}
                        ).astype(bool))
            frame = cv2.imread(os.path.join(video_path, img_file))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return entry, frames, masks


class ReferYouTubeVOSDataloader():
    def __init__(self, args):
        self.data = []
        self.args = args
        print("Loading Refer-Youtube-VOS annotations...")
        self.metadata = json.load(open(DATASET_PATHS[self.args.dataset]["metadata"], 'r'))

        for video_id, video_md in self.metadata["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "video_id": video_id, 
                    "video_path": os.path.join(DATASET_PATHS[self.args.dataset]["video"], video_id), 
                    "mask_path": os.path.join(DATASET_PATHS[self.args.dataset]["masks"], video_id, exp_id),
                    "exp_id": exp_id,
                    "caption": exp_data["exp"],
                    "obj_id": exp_data["obj_id"],
                    "frames": video_md["frames"]
                })
        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_files = sorted([f for f in os.listdir(entry["video_path"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(entry["mask_path"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(image_files) == len(mask_files),\
            f"ERROR: Number of frames ({len(images_files)}) and number of masks ({len(mask_files)}) needs to be equal."
        masks, frames = [], []
        
        for img_file, msk_file in zip(image_files, mask_files):
            # import code; code.interact(local=locals())
            mask = cv2.imread(os.path.join(entry["mask_path"], msk_file), cv2.IMREAD_GRAYSCALE)
            frame = cv2.imread(os.path.join(entry["video_path"], img_file))
            if frame.shape != mask.shape:
                frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))
            masks.append(mask.astype(bool))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return entry, frames, masks

class VideoSegmentationDataloader:
    def __new__(cls, args):
        DATALOADER_MAP = {
            "mevis": MeViSDataloader,
            "rvos": ReferYouTubeVOSDataloader
        }
        return DATALOADER_MAP[args.dataset](args)

# STH_ELSE_REFERRAL_PATH = "/home/we337236/stvg/dataset/something_else/something_else_vidstg_format.json"
# STH_ELSE_FREEFORM_PATH = "/home/c3-0/datasets/stvg/something_else/validation_proc.json" 
# STH_ELSE_VIDEO_PATH = "/home/c3-0/datasets/stvg/something_else/20bn-something-something-v2"

# class SomethingElseLoader:
#     def __init__(self, annotation_path, frame_step=15):
#         with open(annotation_path, 'r') as f:
#             self.data = json.load(f)
#         self.frame_step = frame_step

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         entry = self.data[idx]

#         bbox_data = entry["bbox"]
#         start_frame = entry["st_frame"]
#         end_frame = entry["ed_frame"]

#         cap = cv2.VideoCapture(entry["video_path"])
        
#         frames_with_gt = []
#         sampled_gt_bboxs = []

#         cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        
#         current_frame_idx = start_frame

#         while cap.isOpened():
#             # Stop if we have processed all frames in the annotated range
#             if current_frame_idx > end_frame:
#                 break
            
#             ret, frame = cap.read()
#             # Stop if the video has ended prematurely
#             if not ret:
#                 break
            
#             if (current_frame_idx - start_frame) % self.frame_step == 0:
#                 bbox_sample = bbox_data.get(str(current_frame_idx), [0, 0, 0, 0])
#                 frames_with_gt.append((frame, bbox_sample, current_frame_idx))
#                 sampled_gt_bboxs.append(bbox_sample)
            
#             current_frame_idx += 1

#         cap.release()
#         return frames_with_gt, entry, sampled_gt_bboxs
