"""
VISTA Benchmark Evaluation Script

This script evaluates Vision-Language Models (VLMs) on spatio-temporal video
grounding tasks across multiple datasets. It supports three models (CogVLM, 
Shikra, Ferret) and six datasets (HC-STVG v1/v2, VidSTG, VidVRD, MeViS, RVOS).

The evaluation measures how well models can localize subjects mentioned in
natural language queries across video frames, using mean spatio-temporal IoU
as the primary metric.

Usage:
    python run_eval.py --dataset <dataset> --model <model> --task_type <task> --output_path <path>

Example:
    python run_eval.py --dataset hcstvg1 --model cogvlm --task_type freeform \
        --output_path results/cogvlm_hcstvg1.json

Author: Alejandro Aparcedo

Supported Configurations:
    Models: cogvlm, shikra, ferret
    Datasets: hcstvg1, hcstvg2, vidstg, vidvrd, mevis, rvos
    Tasks: freeform, referral (not required for mevis/rvos)

Output Format:
    JSON file containing:
    - evaluation_parameters: Configuration used for evaluation
    - timing_summary: Performance timing statistics
    - overall_results: Aggregate metrics (m_vIoU, m_vIoU@0.3, m_vIoU@0.5)
    - results: Per-sample predictions and metrics

Notes:
    - All bounding boxes are converted to [xmin, ymin, xmax, ymax] format internally
    - Models output boxes in [0, 1000] normalized space, rescaled to pixel coords
    - Frame sampling is controlled by --frame_step argument
"""

import argparse
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import time

# Local imports
from VISTA.datasets import STVGDataLoader
from VISTA.utils import (
    AverageMeter, 
    Summary, 
    convert_to_python_types, 
    rescale_box_from_1000px, 
    calculate_iou_corners
)
from VISTA.models import FerretSingleSample, ShikraSingleSample, CogVLMSingleSample


def evaluate_entry(frames_with_gt, entry, runner):
    """
    Evaluates a single video entry for HC-STVG, VidVRD, and VidSTG datasets.
    
    Processes each sampled frame through the model, compares predicted bounding
    boxes with ground truth, and computes IoU metrics.
    
    Args:
        frames_with_gt (list): List of tuples (frame, gt_bbox, frame_id)
            - frame: numpy array of the video frame (BGR format)
            - gt_bbox: Ground truth bounding box [xmin, ymin, xmax, ymax]
            - frame_id: Integer frame index in the original video
        entry (dict): Sample metadata containing:
            - caption: Natural language query describing the target
            - video_path: Path to the source video
            - Other dataset-specific fields
        runner: Model wrapper instance with run_inference method
        
    Returns:
        tuple: Evaluation results containing:
            - mv_iou (float): Mean video IoU across all frames
            - mv_iou_03 (float): Fraction of frames with IoU >= 0.3
            - mv_iou_05 (float): Fraction of frames with IoU >= 0.5
            - predicted_boxes (list): List of predicted boxes per frame
            - total_entry_inference_time (float): Total inference time in seconds
            - avg_frame_inference_time (float): Average per-frame inference time
            - num_frames_processed (int): Number of frames evaluated
            - query (str): The prompt sent to the model
            - responses (list): Raw model responses for each frame
    """
    caption = entry["caption"]
    
    # Extract ground truth boxes from frame data
    sampled_gt_boxes = [bbox for frame, bbox, frame_id in frames_with_gt]

    predicted_boxes = []
    responses = []
    predicted_frame_ids = []
    frame_inference_times = []

    # Process each frame
    for frame, gt_bbox, frame_id in frames_with_gt:
        
        # Skip None frames (can occur due to video decoding issues)
        if frame is None:
            print("\n" + "="*50)
            print(f"!!! WARNING: Found a 'None' frame!")
            print(f"    - Video: {entry.get('video_path', 'N/A')}")
            print(f"    - Frame ID: {frame_id}")
            print("    - Skipping this frame.")
            print("="*50 + "\n")
            continue

        # Get frame dimensions for coordinate rescaling
        try:
            frame_H, frame_W = frame.shape[:2]
        except Exception as e:
            print(f"Error getting shape for frame_id {frame_id}: {e}")
            continue
        
        # Convert BGR to RGB and create PIL image for model input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Run model inference with timing
        start_time = time.time()
        text, boxes, query, response = runner.run_inference(pil_image, caption) 
        end_time = time.time()
        print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

        responses.append(response)
        frame_inference_times.append(end_time - start_time)
        
        # Process predicted box (default to empty box if no prediction)
        frame_box = [0, 0, 0, 0]
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].cpu().numpy().flatten()
            # Rescale from [0, 1000] normalized space to pixel coordinates
            frame_box = rescale_box_from_1000px(box, frame_W, frame_H)
        
        predicted_boxes.append(frame_box)
        predicted_frame_ids.append(frame_id)
    
    # Calculate IoU metrics
    frame_ious = [
        calculate_iou_corners(pred, gt) 
        for pred, gt in zip(predicted_boxes, sampled_gt_boxes)
    ]
    mv_iou = np.mean(frame_ious) if frame_ious else 0.0
    mv_iou_03 = np.mean([1 if iou >= 0.3 else 0 for iou in frame_ious])
    mv_iou_05 = np.mean([1 if iou >= 0.5 else 0 for iou in frame_ious])

    # Calculate timing metrics
    total_entry_inference_time = sum(frame_inference_times)
    avg_frame_inference_time = np.mean(frame_inference_times) if frame_inference_times else 0.0
    num_frames_processed = len(frames_with_gt)

    return (
        mv_iou, 
        mv_iou_03, 
        mv_iou_05, 
        predicted_boxes, 
        total_entry_inference_time, 
        avg_frame_inference_time, 
        num_frames_processed, 
        query, 
        responses
    )


def evaluate_entry_mevis_rvos(frames_np, entry, runner):
    """
    Evaluates a single video entry for MeViS and RVOS datasets.
    
    These datasets have a different data format (pre-extracted frames as numpy
    arrays with frame_id-indexed ground truth boxes), requiring a separate
    evaluation function.
    
    Args:
        frames_np (list): List of RGB frame arrays (numpy)
        entry (dict): Sample metadata containing:
            - caption: Natural language query
            - gt_bboxs: Dictionary mapping frame_id to ground truth boxes
            - video_id: Video identifier
        runner: Model wrapper instance with run_inference method
        
    Returns:
        tuple: (metrics, predictions, prompt)
            - metrics (dict): Contains mv_iou, mv_iou03, mv_iou05
            - predictions (dict): Contains pred, pred_boxes, pred_inference_time
            - prompt (str): The prompt sent to the model
    """
    caption = entry["caption"]
    predictions = {
        "pred": [],           # Raw model responses
        "pred_boxes": [],     # Predicted bounding boxes
        "pred_inference_time": [],  # Per-frame inference times
    }

    # Verify frame and annotation counts match
    assert len(frames_np) == len(entry["gt_bboxs"]), \
        f"ERROR: Number of sampled frames ({len(frames_np)}) should match " \
        f"number of sampled GT bboxs ({len(entry['gt_bboxs'])})"
    
    # Process each frame
    for frame, (frame_id, bbox) in zip(frames_np, entry["gt_bboxs"].items()):
        frame_H, frame_W = frame.shape[:2]
        
        # Convert to RGB PIL image (frames are already RGB from dataloader)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Run model inference with timing
        start_time = time.time()
        text, boxes, prompt, response = runner.run_inference(pil_image, caption) 
        end_time = time.time()
        print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

        # Process predicted box
        pred_box = [0, 0, 0, 0]  # Default empty box
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].cpu().numpy().flatten()
            pred_box = rescale_box_from_1000px(box, frame_W, frame_H)

        predictions["pred"].append(response)
        predictions["pred_boxes"].append(pred_box)
        predictions["pred_inference_time"].append(end_time - start_time)

    # Calculate IoU metrics
    gt_boxes = [bbox for frame_id, bbox in entry["gt_bboxs"].items()]
    frame_ious = [
        calculate_iou_corners(pred, gt) 
        for pred, gt in zip(predictions["pred_boxes"], gt_boxes)
    ]
    
    metrics = {
        "mv_iou": np.mean(frame_ious) if frame_ious else 0.0,
        "mv_iou03": np.mean([1 if iou >= 0.3 else 0 for iou in frame_ious]),
        "mv_iou05": np.mean([1 if iou >= 0.5 else 0 for iou in frame_ious])
    }

    return metrics, predictions, prompt


def main(args):
    """
    Main evaluation loop.
    
    Loads the specified dataset and model, runs evaluation on all samples
    (or a subset if specified), and saves results to a JSON file.
    
    Args:
        args: Parsed command-line arguments containing:
            - dataset: Dataset name
            - model: Model name
            - task_type: Task type (referral/freeform)
            - output_path: Output JSON file path
            - device: Compute device
            - entry_index: Starting sample index
            - max_iters: Maximum samples to process
            - frame_step: Frame sampling interval
    """
    # Initialize dataset
    dataset = STVGDataLoader(args)
    
    # Initialize model
    print(f"Initializing model: {args.model}")
    if args.model == 'cogvlm':
        runner = CogVLMSingleSample()
    elif args.model == 'shikra':
        runner = ShikraSingleSample()
    elif args.model == 'ferret':
        runner = FerretSingleSample()
    else:
        raise ValueError(
            f"Model '{args.model}' is not supported. "
            f"Choose from 'cogvlm', 'shikra', or 'ferret'."
        )
    
    # Initialize metric trackers
    mv_iou = AverageMeter('Mean Video IoU', fmt=':.4f', summary_type=Summary.AVERAGE)
    mv_iou03 = AverageMeter('Video IoU@03', fmt=':.4f', summary_type=Summary.AVERAGE)
    mv_iou05 = AverageMeter('Video IoU@05', fmt=':.4f', summary_type=Summary.AVERAGE)
    inferences_times = AverageMeter('Inference_Time', fmt=':.4f', summary_type=Summary.AVERAGE)
    predictions_list = []

    overall_start_time = time.time()
    
    # Determine sample range to evaluate
    if args.entry_index > 0:
        # Start from specified index
        start_index = args.entry_index
        end_index = min(args.entry_index + args.max_iters, len(dataset))
    elif args.max_iters > 0:
        # Limit number of samples
        start_index = 0
        end_index = min(args.max_iters, len(dataset))
    else:
        # Evaluate entire dataset
        start_index = 0
        end_index = len(dataset)

    # Main evaluation loop
    for i in tqdm(range(start_index, end_index), desc="Evaluating..."):

        # Handle MeViS and RVOS datasets (different data format)
        if args.dataset == 'mevis' or args.dataset == 'rvos':
            frames, entry = dataset[i]
            assert len(frames) > 0, "No frames sampled"
            assert len(entry["caption"]) > 0, "Empty caption"
            
            metrics, predictions, prompt = evaluate_entry_mevis_rvos(frames, entry, runner)

            # Update metric trackers
            mv_iou.update(metrics["mv_iou"], len(predictions["pred"]))
            mv_iou03.update(metrics["mv_iou03"], len(predictions["pred"]))
            mv_iou05.update(metrics["mv_iou05"], len(predictions["pred"]))
            inferences_times.update(
                np.mean(predictions["pred_inference_time"]), 
                len(predictions["pred"])
            )

            result = {
                "entry": entry,
                "prompt": prompt,
                "predictions": predictions,
                "metrics": metrics,
            }
            
        else:
            # Handle HC-STVG, VidSTG, VidVRD datasets
            frames_with_gt, entry, gt_bboxs = dataset[i]
            assert len(frames_with_gt) > 0, "No frames sampled"
            assert len(entry["caption"]) > 0, "Empty caption"
            
            (
                entry_mv_iou, 
                mv_iou_03, 
                mv_iou_05, 
                pred_boxes, 
                entry_time, 
                avg_frame_time, 
                num_frames, 
                query, 
                responses
            ) = evaluate_entry(frames_with_gt, entry, runner)
        
            # Convert numpy types to Python types for JSON serialization
            pred_boxes = convert_to_python_types(pred_boxes)

            # Update metric trackers
            mv_iou.update(entry_mv_iou, 1)
            mv_iou03.update(mv_iou_03, 1)
            mv_iou05.update(mv_iou_05, 1)
            inferences_times.update(entry_time, 1)

            result = {
                "entry": entry,
                "queries": query,
                "responses": responses,
                "ground_truth_boxes": gt_bboxs,
                "predicted_boxes": pred_boxes,
                "mvIoU": float(entry_mv_iou),
                "timing_info": {
                    "total_inference_time_seconds": float(entry_time),
                    "frames_processed": int(num_frames),
                    "mean_inference_time_per_frame_seconds": float(avg_frame_time)
                }
            }
            
        predictions_list.append(result)

    # Compute timing summary
    timing_summary = {
        "total_evaluation_time_seconds": float(time.time() - overall_start_time),
        "total_model_inference_time_seconds": float(inferences_times.sum),
        "total_samples_processed": int(end_index - start_index),
        "total_frames_processed": int(mv_iou.count),
        "mean_processing_time_per_sample_seconds": float(inferences_times.avg),
        "mean_inference_time_per_frame_seconds": (
            float(inferences_times.sum / mv_iou.count) if mv_iou.count > 0 else 0
        )
    }

    # Print results summary
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

    # Prepare final output
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

    # Save results to JSON file
    if len(os.path.dirname(args.output_path)) > 0:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Done! Predictions saved to {args.output_path}")


# --------------------------
# Argument Parsing
# --------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="VISTA Benchmark Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate CogVLM on HC-STVG v1 with freeform queries
    python run_eval.py --dataset hcstvg1 --model cogvlm --task_type freeform \\
        --output_path results/cogvlm_hcstvg1_freeform.json

    # Evaluate Shikra on MeViS (no task_type needed)
    python run_eval.py --dataset mevis --model shikra \\
        --output_path results/shikra_mevis.json

    # Debug mode: process only 2 samples starting from index 0
    python run_eval.py --dataset vidstg --model ferret --task_type referral \\
        --output_path results/debug.json --entry_index 0 --max_iters 2
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=['hcstvg1', 'hcstvg2', 'vidstg', 'vidvrd', 'mevis', 'rvos'],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=['cogvlm', 'ferret', 'shikra'],
        help="Model to evaluate"
    )
    parser.add_argument(
        "--task_type", 
        type=str, 
        required=False, 
        choices=['referral', 'freeform'],
        help="Query type: 'referral' (template-based) or 'freeform' (natural language). "
             "Required for hcstvg, vidstg, vidvrd datasets."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path for output JSON file with predictions and metrics"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--entry_index", 
        type=int, 
        default=-1,
        help="Starting sample index. Use >= 0 to start from specific sample, "
             "-1 for beginning of dataset (default: -1)"
    )
    parser.add_argument(
        "--frame_step", 
        type=int, 
        default=5,
        help="Frame sampling interval - process every Nth frame (default: 5)"
    )
    parser.add_argument(
        "--max_iters", 
        type=int, 
        default=-1,
        help="Maximum number of samples to process. Use positive value for "
             "testing/debugging, -1 for full dataset (default: -1)"
    )
    
    args = parser.parse_args()
    print(f"\nEvaluation Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Task Type: {args.task_type}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Frame Step: {args.frame_step}")
    print()
    
    main(args)