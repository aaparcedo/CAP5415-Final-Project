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
from transformers import AutoTokenizer, CLIPImageProcessor

# --- VideoLISA Imports ---
from VideoLISA.model.VideoLISA import VideoLISAForCausalLM
from VideoLISA.model.llava import conversation as conversation_lib
from VideoLISA.model.llava.mm_utils import tokenizer_image_token
from VideoLISA.model.segment_anything.utils.transforms import ResizeLongestSide
from VideoLISA.utils.utils import DEFAULT_IMAGE_TOKEN
from VideoLISA.evaluation.refdavis.davis2017.metrics import db_eval_iou, db_eval_boundary

from enum import Enum
import numpy as np
import torch
import torch.distributed as dist

from datasets import VideoSegmentationDataloader
from utils import Summary, AverageMeter, encode_masks_to_rle

DEVICE = "cuda"

class VideoLISARunner:
    """
    A class to handle loading the VideoLISA model and running inference.
    """
    def __init__(self):
        print("Loading VideoLISA model...")
        self.image_size = 1024
        self.model_max_length = 512
        self.vision_tower = "openai/clip-vit-large-patch14-336"
        self.precision = "bf16"
        self.conv_type = "phi3_instruct"
        self.transform = ResizeLongestSide(self.image_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "ZechenBai/VideoLISA-3.8B", model_max_length=self.model_max_length, padding_side="right", use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

        torch_dtype = torch.bfloat16

        self.model = VideoLISAForCausalLM.from_pretrained(
            "ZechenBai/VideoLISA-3.8B",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            vision_tower=self.vision_tower,
            seg_token_idx=self.seg_token_idx,
            attn_implementation="flash_attention_2",
        ).to(DEVICE)

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower_handle = self.model.get_model().get_vision_tower()
        vision_tower_handle.to(dtype=torch_dtype, device=DEVICE)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower)
        self.model.eval()
        print("VideoLISA model loaded successfully!")

    def _preprocess_sam(self, x):
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(x.device, x.dtype)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(x.device, x.dtype)
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def run_inference(self, frames_np: list, caption: str, num_frames_sparse=32, num_frames_dense=4):
        total_frames = len(frames_np)
        conv = conversation_lib.conv_templates[self.conv_type].copy()
        conv.messages = []
        prompt = caption.lower()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "Sure, [SEG].")
        prompt_str = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_str, self.tokenizer, return_tensors="pt").unsqueeze(0).to(DEVICE)

        if total_frames >= num_frames_sparse:
            sparse_idxs = [int(i) for i in np.linspace(0, total_frames - 1, num_frames_sparse)]
        else:
            sparse_idxs = list(range(total_frames)) + [total_frames - 1] * (num_frames_sparse - total_frames)
        
        dense_idxs_choice = np.linspace(0, num_frames_sparse - 1, num_frames_dense, dtype=int)
        valid_dense_idxs = [dense_idxs_choice[i] for i in range(num_frames_dense)]

        image_list_sam, image_list_clip = [], []
        original_size_list, resize_list = [frames_np[0].shape[:2]], []

        for frm_idx in sparse_idxs:
            image_clip = self.clip_image_processor.preprocess(frames_np[frm_idx], return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            image_list_clip.append(image_clip)

        for image_np in frames_np:
            image_sam_resized = self.transform.apply_image(image_np)
            if not resize_list:
                resize_list.append(image_sam_resized.shape[:2])
            image_sam = self._preprocess_sam(torch.from_numpy(image_sam_resized).permute(2, 0, 1).contiguous()).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            image_list_sam.append(image_sam)

        image_sam_tensor = torch.stack(image_list_sam, dim=1)
        image_clip_tensor = torch.stack(image_list_clip, dim=1)

        with torch.inference_mode():
            _, pred_masks_tensors = self.model.evaluate(
                image_clip_tensor, image_sam_tensor, input_ids, resize_list, original_size_list, dense_indices=[valid_dense_idxs]
            )

        pred_masks = pred_masks_tensors[0].cpu().numpy() > 0
        return pred_masks, prompt

# ===================================================================================
# Evaluation Entry Functions
# ===================================================================================

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
    runner = VideoLISARunner()    
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
            "total_samples_processed": int(end_index-start_index),
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

# ===================================================================================
# Argument Parsing
# ===================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoLISA Evaluation on MeViS or Refer-YouTube-VOS")
    parser.add_argument("--dataset", type=str, required=True, choices=['mevis', 'rvos'], help="The dataset to evaluate on.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for the output JSON file.")
    parser.add_argument("--entry_index", type=int, default=-1, help="Single entry index to test, or -1 for full run.")
    parser.add_argument("--max_iters", type=int, default=-1, help="Max number of videos to process (-1 for all).")
    
    args = parser.parse_args()
    print(args)
    main(args)