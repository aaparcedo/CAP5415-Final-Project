# ------------------------------------------
# Script to evaluate 
# models: CogVLM, Ferret-V1, and Shikra 
# datasets: HC-STVG-V1, HC-STVG-V2, Something Else, VidSTG, and VidVRD
# tasks: freeform, referral
# 
# important things
# - all bounding boxes are converted to [xmin, ymin, xmax, ymax], try to maintain this consistency
# - HC-STVG-V1, VidSTG, and VidVRD format: [x, y, w, h]
# - Something Else format: [xmin, ymin, xmax, ymax]
# ------------------------------------------
import argparse
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

# Dataset imports
from datasets import STVGDataLoader
from utils import AverageMeter, Summary, convert_to_python_types, rescale_box_from_1000px, calculate_iou_corners
from models import FerretSingleSample, ShikraSingleSample, CogVLMSingleSample


# Evaluation function for HC-STVG-1&2, VidVRD, and VidSTG
def evaluate_entry(frames_with_gt, entry, runner):
    caption = entry["caption"]

    sampled_gt_boxes = [bbox for frame, bbox, frame_id in frames_with_gt]

    predicted_boxes = []
    responses = []
    predicted_frame_ids = []
    frame_inference_times = []

    for frame, gt_bbox, frame_id in frames_with_gt:
        

        if frame is None:
            print("\n\n" + "="*50)
            print(f"!!! CONFIRMED: Found a 'None' frame!")
            print(f"    - Video: {entry.get('video_path', 'N/A')}")
            print(f"    - Frame ID in Annotation: {frame_id}")
            print(f"    - Index in the list of frames: {i}")
            print("    - This is the frame that was causing the crash.")
            print("="*50 + "\n")
            continue # Skips this bad frame and continues to the next

        try:
            frame_H, frame_W = frame.shape[:2]
        except Exception as e:
            print(f"Error getting shape for frame_id {frame_id} even after None check. Error: {e}")
            continue
                
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        start_time = time.time()
        text, boxes, query, response = runner.run_inference(pil_image, caption) 
        end_time = time.time()
        print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

        responses.append(response)

        frame_inference_times.append(end_time - start_time)
        frame_box = [0, 0, 0, 0] # Default box
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].cpu().numpy().flatten()

            frame_box = rescale_box_from_1000px(box, frame_W, frame_H)
        
        predicted_boxes.append(frame_box)
        predicted_frame_ids.append(frame_id)
        
    frame_ious = [calculate_iou_corners(pred, gt) for pred, gt in zip(predicted_boxes, sampled_gt_boxes)]
    mv_iou = np.mean(frame_ious) if frame_ious else 0.0

    mv_iou_03 = np.mean([1 if iou >= 0.3 else 0 for iou in frame_ious])
    mv_iou_05 = np.mean([1 if iou >= 0.5 else 0 for iou in frame_ious])

    # Calculate timing metrics for this entry
    total_entry_inference_time = sum(frame_inference_times)
    avg_frame_inference_time = np.mean(frame_inference_times) if frame_inference_times else 0.0
    num_frames_processed = len(frames_with_gt)

    return mv_iou, mv_iou_03, mv_iou_05, predicted_boxes, total_entry_inference_time, avg_frame_inference_time, num_frames_processed, query, responses

def evaluate_entry_mevis_rvos(frames_np, entry, runner):
    caption = entry["caption"]
    predictions = {
        "pred": [],
        "pred_boxes": [],
        "pred_inference_time": [],
    }

    assert len(frames_np) == len(entry["gt_bboxs"]), \
        f"ERROR: Number of sampled frames ({len(frames_np)}) should match number of sampled GT bboxs ({len(entry['gt_bboxs'])})"
    
    for frame, (frame_id, bbox) in zip(frames_np, entry["gt_bboxs"].items()):
        frame_H, frame_W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        start_time = time.time()
        text, boxes, prompt, response = runner.run_inference(pil_image, caption) 
        end_time = time.time()
        print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

        pred_box = [0, 0, 0, 0] # Default box
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].cpu().numpy().flatten()
            pred_box = rescale_box_from_1000px(box, frame_W, frame_H)

        predictions["pred"].append(response)
        predictions["pred_boxes"].append(pred_box)
        predictions["pred_inference_time"].append(end_time - start_time)

    frame_ious = [calculate_iou_corners(pred, gt) for pred, gt in zip(predictions["pred_boxes"], [bbox for frame_id, bbox in entry["gt_bboxs"].items()])]
    metrics = {
        "mv_iou": np.mean(frame_ious) if frame_ious else 0.0,
        "mv_iou03": np.mean([1 if iou >= 0.3 else 0 for iou in frame_ious]),
        "mv_iou05": np.mean([1 if iou >= 0.5 else 0 for iou in frame_ious])
    }

    return metrics, predictions, prompt


def main(args):
    dataset = STVGDataLoader(args)
    print(f"Initializing model: {args.model}")
    if args.model == 'cogvlm':
        runner = CogVLMSingleSample()
    elif args.model == 'shikra':
        runner = ShikraSingleSample()
    elif args.model == 'ferret':
        runner = FerretSingleSample()
    else:
        raise ValueError(f"Model '{args.model}' is not supported. Choose from 'cogvlm', 'shikra', or 'ferret' .")
        
    mv_iou = AverageMeter('Mean Video IoU', fmt=':.4f', summary_type=Summary.AVERAGE)
    mv_iou03 = AverageMeter('Video IoU@03', fmt=':.4f', summary_type=Summary.AVERAGE)
    mv_iou05 = AverageMeter('Video IoU@05', fmt=':.4f', summary_type=Summary.AVERAGE)
    inferences_times = AverageMeter('Inference_Time', fmt=':.4f', summary_type=Summary.AVERAGE)
    predictions_list = []

    overall_start_time = time.time()
    
    if args.entry_index > 0:
        start_index, end_index = args.entry_index, min(args.entry_index + args.max_iters, len(dataset))
    elif args.max_iters > 0:
        start_index, end_index = 0, min(args.max_iters, len(dataset))
    else:
        start_index, end_index = 0, len(dataset)

    
    for i in tqdm(range(start_index, end_index), desc="Evaluating..."):

        # try:
        # import code; code.interact(local=locals())
        if args.dataset == 'mevis' or args.dataset == 'rvos':
            frames, entry = dataset[i]
            assert len(frames) > 0; "no frames sampled"
            assert len(entry["caption"]) > 0; "empty caption"
            metrics, predictions, prompt = evaluate_entry_mevis_rvos(frames, entry, runner)

            mv_iou.update(metrics["mv_iou"], len(predictions["pred"]))
            mv_iou03.update(metrics["mv_iou03"], len(predictions["pred"]))
            mv_iou05.update(metrics["mv_iou05"], len(predictions["pred"]))
            inferences_times.update(np.mean(predictions["pred_inference_time"]), len(predictions["pred"]))

            result = {
                "entry": entry,
                "prompt": prompt,
                "predictions": predictions,
                "metrics": metrics,
            }
            
        else:
            frames_with_gt, entry, gt_bboxs = dataset[i]
            assert len(frames_with_gt) > 0; "no frames sampled"
            assert len(entry["caption"]) > 0; "empty caption"
            mv_iou, mv_iou_03, mv_iou_05, pred_boxes, entry_time, avg_frame_time, num_frames, query, responses = evaluate_entry(frames_with_gt, entry, runner, args.dataset)
        
            pred_boxes = convert_to_python_types(pred_boxes)

            mv_iou.update(mv_iou, 1)
            mv_iou03.update(mv_iou_03, 1)
            mv_iou05.update(mv_iou_05, 1)
            inferences_times.update(timing_info["total_inference_time"], 1)

            result = {
                "entry": entry,
                "queries": query,
                "responses": responses,
                "ground_truth_boxes": gt_bboxs,
                "predicted_boxes": pred_boxes,
                "mvIoU": float(mv_iou),
                "timing_info": {
                    "total_inference_time_seconds": float(entry_time),
                    "frames_processed": int(num_frames),
                    "mean_inference_time_per_frame_seconds": float(avg_frame_time)
                }
            }
        predictions_list.append(result)
    
        # except Exception as e:
        #     frames_with_gt, entry, gt_bboxs = dataset[i]
        #     print(f"\n--- CRITICAL ERROR ---\n"
        #             f"Failed to process entry index {i} (video: {entry['video_path']}).\n"
        #             f"Error: {e}\n"
        #             f"Skipping and continuing.\n"
        #             f"----------------------\n")
        #     mv_iou.update(0, len(dataset[i]["gt_bboxs"]))
        #     mv_iou03.update(0, len(dataset[i]["gt_bboxs"]))
        #     mv_iou05.update(0, len(dataset[i]["gt_bboxs"]))
        #     predictions_list.append({
        #         "entry": dataset[i],
        #         "status": "PROCESSING FAILED. CRITICAL ERROR. SKIPPING.",
        #         "reason": f"{type(e).__name__}: {e}",
        #         "metrics": {
        #             "mv_iou": 0,
        #             "mv_iou03": 0,
        #             "mv_iou05": 0
        #         }
        #     })
        #     continue

    timing_summary = {
        "total_evaluation_time_seconds": float(time.time() - overall_start_time),
        "total_model_inference_time_seconds": float(inferences_times.sum),
        "total_samples_processed": int(len(dataset)),
        "total_frames_processed": int(mv_iou.count),
        "mean_processing_time_per_sample_seconds": float(inferences_times.avg),
        "mean_inference_time_per_frame_seconds": float(inferences_times.sum / mv_iou.count) if mv_iou.count > 0 else 0
    }

    print("\n--- Evaluation Results ---")
    print(mv_iou.summary())
    print(mv_iou03.summary())
    print(mv_iou05.summary())

    print("\n--- Timing Summary ---")
    print(f"Total Evaluation Time: {timing_summary['total_evaluation_time_seconds']:.2f} seconds")
    print(f"Total Model Inference Time: {timing_summary['total_model_inference_time_seconds']:.2f} seconds")
    print(f"Total Samples Processed: {end_index - start_index}")
    print(f"Total Frames Processed: {timing_summary['total_frames_processed']}")
    print(f"Mean Processing Time per Sample: {timing_summary['mean_processing_time_per_sample_seconds']:.4f} seconds")
    print(f"Mean Inference Time per Frame: {timing_summary['mean_inference_time_per_frame_seconds']:.4f} seconds")
    print("----------------------")

    final_output = {
        "evaluation_parameters": {
            "frame_step": args.frame_step if args.dataset != 'rvos' else 0,
            "dataset": args.dataset,
            "model": args.model,
            "task_type": args.task_type,
        },
        "timing_summary": timing_summary,
        "overall_results": {
            "avg_mviou": mv_iou.avg,
            "avg_mviou03": mv_iou03.avg,
            "avg_mviou05": mv_iou05.avg,
        },
        "results": predictions_list
    }

    if len(os.path.dirname(args.output_path)) > 0: os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Done! Predictions saved to {args.output_path}")

# --------------------------
# Argument Parsing and Execution
# --------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="CogVLM Evaluation on HC-STVG Dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="select one from ('hcstvg1', 'hcstvg2', 'sthelse', 'vidvrd', 'vidstg')")
    parser.add_argument("--model", type=str, required=True,
                        help="select one from ('cogvlm', 'ferret', 'shikra')")
    parser.add_argument("--task_type", type=str, required=False, choices=['referral', 'freeform'],
                        help="Task type i.e., 'referral', 'freeform'")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path of output JSON file.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference (default: cuda)")
    parser.add_argument("--entry_index", type=int, default=-1,
                        help="Index of a single entry to test (>= 0) or -1 for full dataset evaluation")
    parser.add_argument("--frame_step", type=int, default=5,
                        help="Frame sampling step")
    parser.add_argument("--max_iters", type=int, default=-1,
                        help="Maximum number of iterations (batches) to process (use a positive value for testing, -1 for full dataset)")
    
    args = parser.parse_args()
    print(args)
    main(args)

