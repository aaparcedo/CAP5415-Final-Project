"""
Dataset Loaders for VISTA Benchmark

This module provides dataset loader classes for various spatio-temporal video grounding
datasets used in the VISTA benchmark evaluation. Each loader handles video frame extraction,
bounding box annotation loading, and query (caption) retrieval.

Supported Datasets:
    - HC-STVG v1 & v2: Human-centric spatio-temporal video grounding
    - VidSTG: Video spatio-temporal grounding
    - VidVRD: Video visual relation detection
    - MeViS: Motion expressions video segmentation
    - RVOS: Referring YouTube-VOS

Author: Alejandro Aparcedo
"""

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
from .utils import xywh_to_corners
from pycocotools import mask as mask_util


# --------------------------
# CONFIGURATION
# --------------------------
DEVICE = "cuda"
CAPTION_MOD_TEST1_PATH = "/home/aparcedo/IASEB/test1_caption_modification_hcstvg1.json"

# Dataset paths configuration
# NOTE: Update these paths to match your local data directory structure
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
    """
    Dataloader for HC-STVG (Human-Centric Spatio-Temporal Video Grounding) datasets.
    
    Supports both HC-STVG v1 and v2 datasets with referral and freeform query types.
    Loads video frames and corresponding bounding box annotations for evaluation.
    
    Attributes:
        referral_data_path (str): Path to referral caption annotations
        video_dir (str): Directory containing video files
        frame_step (int): Interval for frame sampling
        data (list): Loaded annotation data
        
    Args:
        args: Argument namespace containing:
            - dataset: Dataset name ('hcstvg1' or 'hcstvg2')
            - task_type: Query type ('referral' or 'freeform')
            - frame_step: Frame sampling interval
    """
    
    def __init__(self, args):
        self.referral_data_path = DATASET_PATHS[args.dataset]["referral"]
        self.video_dir = DATASET_PATHS[args.dataset]["video"]
        self.frame_step = args.frame_step
        self.args = args
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        if self.args.task_type == 'referral':
            self.referral_caption_data = json.load(open(self.referral_data_path, 'r'))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (frames_with_gt, entry, sampled_gt_bboxs)
                - frames_with_gt: List of (frame, bbox, frame_id) tuples
                - entry: Dictionary containing sample metadata and caption
                - sampled_gt_bboxs: List of ground truth bounding boxes
        """
        entry = self.data[idx]
        video_path = os.path.join(self.video_dir, entry['video_path'])
        cap = cv2.VideoCapture(video_path)

        # Update caption for referral task type
        if self.args.task_type == 'referral':
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
        current_frame_idx = start_frame

        # Extract frames at specified intervals
        while cap.isOpened():
            if current_frame_idx > end_frame:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames according to frame_step
            if (current_frame_idx - start_frame) % self.frame_step == 0:
                box_index = current_frame_idx - start_frame
                if 0 <= box_index < len(bbox_data):
                    bbox = xywh_to_corners(bbox_data[box_index])
                    frames_with_gt.append((frame, bbox, current_frame_idx))
                    sampled_gt_bboxs.append(bbox)
            
            current_frame_idx += 1

        cap.release()
        return frames_with_gt, entry, sampled_gt_bboxs


class VidVRDDataloader:
    """
    Dataloader for VidVRD (Video Visual Relation Detection) dataset.
    
    Handles loading of video frames and bounding box annotations in the VidVRD format.
    Ground truth boxes are stored in [x, y, w, h] format and converted to 
    [xmin, ymin, xmax, ymax] during loading.
    
    Attributes:
        data (list): Loaded annotation data
        frame_step (int): Interval for frame sampling
        
    Args:
        args: Argument namespace containing dataset and frame_step parameters
    """
    
    def __init__(self, args):
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        self.frame_step = args.frame_step

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (frames_with_gt, entry, sampled_gt_bboxs)
        """
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


class VidSTGDataloader:
    """
    Dataloader for VidSTG (Video Spatio-Temporal Grounding) dataset.
    
    Similar structure to VidVRD but with VidSTG-specific annotation format.
    
    Attributes:
        data (list): Loaded annotation data
        frame_step (int): Interval for frame sampling
        
    Args:
        args: Argument namespace containing dataset and frame_step parameters
    """
    
    def __init__(self, args):
        self.data = json.load(open(DATASET_PATHS[args.dataset][args.task_type], 'r'))
        self.frame_step = args.frame_step

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (frames_with_gt, entry, sampled_gt_bboxs)
        """
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


class MeViSBBoxDataloader:
    """
    Dataloader for MeViS dataset with bounding box annotations.
    
    Loads frames and ground truth bounding boxes at specified intervals.
    Used for evaluation with bounding box predictions rather than segmentation masks.
    
    Attributes:
        data (list): List of video/expression pairs with annotations
        args: Configuration arguments
        boxes (dict): Bounding box annotations
        metadata (dict): Video and expression metadata
        
    Args:
        args: Argument namespace containing dataset and frame_step parameters
    """
    
    def __init__(self, args):
        print("Loading MeViS annotations...")
        self.data = []
        self.args = args
        self.boxes = json.load(open(DATASET_PATHS[args.dataset]["bbox"], 'r'))
        self.metadata = json.load(open(DATASET_PATHS[args.dataset]["metadata"], 'r'))

        # Build list of video/expression pairs
        for video_id, video_md in self.metadata["videos"].items():
            for exp_id, exp_data in video_md["expressions"].items():
                self.data.append({
                    "video_id": video_id,
                    "video_path": os.path.join(DATASET_PATHS[args.dataset]["video"], video_id), 
                    "caption": exp_data["exp"],
                    "anno_id": exp_data["anno_id"][0],
                    "obj_id": exp_data["obj_id"][0],
                    "gt_bboxs": {
                        frame_id: xywh_to_corners(bbox) 
                        for frame_id, bbox in self.boxes["videos"][video_id]["expressions"][exp_id]["trajectory"].items()
                    },
                })

        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (all_frames_np, entry)
                - all_frames_np: List of RGB frame arrays
                - entry: Dictionary with metadata and sampled ground truth boxes
        """
        entry = self.data[idx]
        video_path = entry["video_path"]
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


class ReferYouTubeVOSBBoxDataloader:
    """
    Dataloader for Refer-YouTube-VOS dataset with bounding box annotations.
    
    Loads video frames and corresponding bounding box annotations for the
    referring video object segmentation task, adapted for bounding box evaluation.
    
    Note: RVOS frames are pre-sampled at frame_step=5, so no additional 
    sampling is performed during loading.
    
    Attributes:
        annotations (dict): Loaded bounding box annotations
        data (list): List of video/expression pairs
        
    Args:
        args: Argument namespace containing dataset parameters
    """
    
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
                    "gt_bboxs": {
                        frame_id: xywh_to_corners(bbox) 
                        for frame_id, bbox in exp_data["trajectory"].items()
                    },
                })
        print(f'Total video/expression pairs loaded: {len(self.data)}')

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (all_frames_np, entry)
        """
        entry = self.data[idx]
        video_path = entry["video_path"]
        image_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        sampled_gt_bboxs = {}

        # RVOS is pre-sampled at frame_step=5; no additional sampling needed
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
    """
    Factory class for creating dataset-specific dataloaders.
    
    Uses the factory pattern to instantiate the appropriate dataloader
    based on the dataset name specified in args.
    
    Supported datasets: hcstvg1, hcstvg2, vidstg, vidvrd, mevis, rvos
    
    Args:
        args: Argument namespace containing 'dataset' field
        
    Returns:
        Dataset-specific dataloader instance
        
    Example:
        >>> args.dataset = 'hcstvg1'
        >>> loader = STVGDataLoader(args)  # Returns HCSTVGDataloader instance
    """
    
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