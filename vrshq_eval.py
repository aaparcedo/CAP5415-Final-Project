# ------------------------------------------
# Unified Script to evaluate VideoLISA on MeViS or Refer-YouTube-VOS
# ------------------------------------------
import argparse
import json
import os
import sys
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPProcessor, AutoConfig

import torchvision.transforms as T

from VideoLISA.evaluation.refdavis.davis2017.metrics import db_eval_iou, db_eval_boundary

from utils import Summary, AverageMeter, encode_masks_to_rle
from datasets import VideoSegmentationDataloader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__name__)), 'VRSHQ'))

# --- VRS-HQ Imports ---
from VRSHQ.model.VISA_multiseg_cliptree_bf16 import VrshqForCausalLM
from VRSHQ.model.llava import conversation as conversation_lib
from VRSHQ.model.llava.mm_utils import tokenizer_image_token
from VRSHQ.model.segment_anything.utils.transforms import ResizeLongestSide
from VRSHQ.dataset.utils import (
    DEFAULT_IMAGE_TOKEN, UNIFIED_SHORT_QUESTION_LIST, ANSWER_LIST
)


# --------------------------
# CONFIGURATION
# --------------------------
DEVICE = "cuda"

DEFAULT_VIDEO_TOKEN = "<video>"
VRSHQ_MODEL_PATH="/home/aparcedo/.cache/huggingface/hub/models--SitongGong--VRS-HQ/snapshots/d5e9d67f25a09a2a223fc3949b3028fc31d7b425"

# ===================================================================================
# VRSHQRunner Class
# ===================================================================================

class VRSHQRunner:
    def __init__(self, model_path=VRSHQ_MODEL_PATH):
        print("Loading VRS-HQ model from:", model_path)
        self.image_size = 1024
        self.model_max_length = 2048
        # self.vision_tower = "openai/clip-vit-large-patch14-336"
        self.vision_tower = "openai/clip-vit-large-patch14"
        self.precision = "bf16"
        self.conv_type = "llava_v1"
        self.transform = ResizeLongestSide(self.image_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=self.model_max_length, padding_side="right", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.tokenizer.add_tokens("[SEG]", special_tokens=True)
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
        self.tokenizer.add_tokens("[TAK]", special_tokens=True)
        self.track_token_idx = self.tokenizer("[TAK]", add_special_tokens=False).input_ids[0]

        assert self.seg_token_idx != self.track_token_idx,\
            f'[SEG] ({self.seg_token_idx}) and [TAK] ({self.track_token_idx}) token indices cannot be the same'

        torch_dtype = torch.bfloat16

        model_args = {
            "train_mask_decoder": False,
            "out_dim": 256,
            "seg_token_idx": self.seg_token_idx,
            "seg_token_num": 1, 
            "track_token_idx": self.track_token_idx,
            "vision_pretrained": "/home/aparcedo/IASEB/VRSHQ/checkpoints/sam2_hiera_large.pt",
            "alpha": 0.1,
            "vision_tower": self.vision_tower,
            "use_im_start_end": False,
        }

        self.model = VrshqForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto", **model_args
        )

        print("Loading fine-tuned segmentation weights...")

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower_handle = self.model.get_model().get_vision_tower()
        vision_tower_handle.to(dtype=torch_dtype, device=DEVICE)

        model_args_from_pt = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_path)
        model_args_from_pt.use_cluster = True
        model_args_from_pt.freeze = False
        model_args_from_pt.mm_tune = True
        model_args_from_pt.spatial_cluster_rate0 = 64
        model_args_from_pt.spatial_cluster_rate1 = 32
        model_args_from_pt.spatial_cluster_rate2 = 16
        model_args_from_pt.temporal_cluster_rate = 0.0625
        model_args_from_pt.use_cluster = True
        model_args_from_pt.vision_tune = False
        self.model.get_model().initialize_cluster_modules(model_args_from_pt)

        state_dict = torch.load(os.path.join(model_path, "pytorch_model-00002-of-00002.bin"), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)     
        
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        self.clip_processor = CLIPProcessor.from_pretrained(self.vision_tower)
        
        self.model.eval()
        self.model.initialize_clip_modules(self.vision_tower)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("VRS-HQ model loaded successfully!")

    def _preprocess_sam(self, x):
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(x.device, x.dtype)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(x.device, x.dtype)
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def run_inference(self, frames_np: list, caption: str):
        conv = conversation_lib.conv_templates[self.conv_type].copy()
        conv.messages = []
        
        conv.append_message(conv.roles[0], UNIFIED_SHORT_QUESTION_LIST[0].format(sent=caption))
        conv.append_message(conv.roles[1], ANSWER_LIST[0])

        prompt_str = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_str, self.tokenizer, return_tensors="pt").to(DEVICE)

        image_list_sam, image_list_clip = [], []
        original_size = frames_np[0].shape[:2] 

        sam_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for image_np in frames_np:
            # Preprocess for SAM
            image_sam_resized = self.transform.apply_image(image_np) # Resize to 1024
            image_sam_tensor = sam_transform(image_sam_resized).to(DEVICE, dtype=self.model.dtype)
            h, w = image_sam_tensor.shape[-2:]
            padh, padw = self.image_size - h, self.image_size - w
            image_sam_padded = F.pad(image_sam_tensor, (0, padw, 0, padh))
            image_list_sam.append(image_sam_padded)
            
            # Preprocess for CLIP
            image_clip_tensor = self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].to(DEVICE, dtype=self.model.dtype)
            image_list_clip.append(image_clip_tensor)

        images_sam_tensor = torch.stack(image_list_sam, dim=0)
        images_clip_tensor = torch.stack(image_list_clip, dim=0)
        
        clip_inputs = self.clip_processor(text=[caption], images=frames_np, return_tensors="pt", padding=True)

        with torch.inference_mode():
            output = self.model.model_forward(
                images=[images_sam_tensor],
                images_clip=[images_clip_tensor],
                input_ids=input_ids,
                labels=None,
                attention_masks=input_ids.ne(self.tokenizer.pad_token_id),
                offset=torch.LongTensor([0, 1]).to(DEVICE),
                masks_list=[],
                label_list=[torch.zeros(original_size)],
                resize_list=[self.transform.apply_image(frames_np[0]).shape[:2]],
                conversation_list=[prompt_str],
                num_frame_list=[len(frames_np)],
                num_conv_list=[1],
                inference=True,
                cond_frame_list=[[0]], 
                clip_input_list=[clip_inputs],
                tokenizer=self.tokenizer,
            )
        
        pred_masks_tensors = output["pred_masks"]
        pred_masks = pred_masks_tensors.cpu().numpy() > 0
        return pred_masks, prompt_str

def evaluate_entry(entry, frames, gt_masks, runner):
    start_time = time.time()
    pred_masks_np, prompt = runner.run_inference(frames, entry["caption"])
    end_time = time.time()
    
    assert len(gt_masks) == len(pred_masks_np), f"ERROR: Number of GT masks ({len(gt_masks)}) must match number of predicted masks ({len(pred_masks_np)})"

    rle_predictions = encode_masks_to_rle(pred_masks_np)
    
    j_scores, f_scores = [], []
    for pred_mask, gt_mask in zip(pred_masks_np, gt_masks):
        j_scores.append(db_eval_iou(gt_mask, pred_mask))
        f_scores.append(db_eval_boundary(gt_mask, pred_mask))

    metrics = {
        "mean_j": np.mean(j_scores),
        "mean_f": np.mean(f_scores),
        "mean_j_and_f": (np.mean(j_scores) + np.mean(f_scores)) / 2
    }

    predictions = {
        "preds_masks_rle": rle_predictions,
        "pred_inference_time": (end_time - start_time),
    }
        
    return metrics, prompt, predictions


def main(args):
    runner = VRSHQRunner()
    dataset = VideoSegmentationDataloader(args)

    start_index = args.entry_index if args.entry_index >= 0 else 0
    end_index = min(start_index + args.max_iters, len(dataset)) if args.max_iters > 0 else len(dataset)

    results_list, failed_samples = [], 0
    overall_start_time = time.time()

    jaccard = AverageMeter('Mean Jaccard (J)', fmt=':.4f', summary_type=Summary.AVERAGE)
    countour = AverageMeter('Mean Countour (F)', fmt=':.4f', summary_type=Summary.AVERAGE)
    jf_score = AverageMeter('Mean J&F Score', fmt=':.4f', summary_type=Summary.AVERAGE)
    inferences_times = AverageMeter('Inference_Time', fmt=':.4f', summary_type=Summary.AVERAGE)

    for i in tqdm(range(start_index, end_index), desc=f"Evaluating on {args.dataset.upper()}..."):
        # try:
        entry, frames, gt_masks = dataset[i]
        assert frames, f"ERROR: No frames in this dataset sample: {entry['video_path']}"
        metrics, prompt, predictions = evaluate_entry(entry, frames, gt_masks, runner)
        jaccard.update(metrics['mean_j'], len(frames))
        countour.update(metrics['mean_f'], len(frames))
        jf_score.update(metrics['mean_j_and_f'], len(frames))
        inferences_times.update(np.mean(predictions["pred_inference_time"]), len(frames))
        
        results_list.append({
            "entry": entry, 
            "prompt": prompt, 
            "predictions": predictions,
            "metrics": metrics,
        })
        # except Exception as e:
        #     failed_samples += 1
        #     results_list.append({
        #         "entry": dataset.data[i],
        #         "status": "PROCESSING FAILED. CRITICAL ERROR. SKIPPING.",
        #         "reason": f"{type(e).__name__}: {e}",
        #         "metrics": {"mean_jaccard": 0, "mean_contour": 0, "mean_j_and_f": 0}})

    print("\n--- Evaluation Results ---")
    print(f"Dataset: {args.dataset.upper()}")
    print(jaccard.summary())
    print(countour.summary())
    print(jf_score.summary())
    print(f"Failed Samples: {failed_samples}")
    print("--------------------------")

    final_output = {
        "evaluation_parameters": vars(args),
        "timing_summary": {
            "total_evaluation_time_seconds": float(time.time() - overall_start_time),
            "total_model_inference_time_seconds": float(inferences_times.sum),
            "total_samples_processed": int(len(dataset)),
            "total_frames_processed": int(jaccard.count),
            "mean_processing_time_per_sample_seconds": float(inferences_times.avg),
            "mean_inference_time_per_frame_seconds": float(inferences_times.sum / inferences_times.count)
        },
        "overall_results": {
            "avg_j": jaccard.avg,
            "avg_f": countour.avg,
            "avg_j_and_f": jf_score.avg
        },
        "results": results_list
    }

    if len(os.path.dirname(args.output_path)) > 0: os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Done! Predictions saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoLISA Evaluation on MeViS or Refer-YouTube-VOS")
    parser.add_argument("--dataset", type=str, required=True, choices=['mevis', 'rvos'], help="The dataset to evaluate on.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for the output JSON file.")
    parser.add_argument("--entry_index", type=int, default=-1, help="Single entry index to test, or -1 for full run.")
    parser.add_argument("--max_iters", type=int, default=-1, help="Max number of videos to process (-1 for all).")
    
    args = parser.parse_args()
    print(args)
    main(args)