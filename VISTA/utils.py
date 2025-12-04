"""
Utility Functions for VISTA Benchmark

This module provides utility functions for the VISTA benchmark evaluation,
including bounding box operations, IoU calculation, metric tracking, and
video visualization.

Author: Alejandro Aparcedo
"""

from pycocotools import mask as mask_util
from enum import Enum
import numpy as np
import torch
import torch.distributed as dist
import cv2


def calculate_iou_corners(box1, box2):
    """
    Computes Intersection over Union (IoU) for two bounding boxes.
    
    Both boxes should be in corner format: [xmin, ymin, xmax, ymax].
    Used for evaluating spatial overlap between predicted and ground truth boxes.
    
    Args:
        box1 (list): First bounding box [xmin, ymin, xmax, ymax]
        box2 (list): Second bounding box [xmin, ymin, xmax, ymax]
        
    Returns:
        float: IoU value between 0 and 1
        
    Example:
        >>> box1 = [10, 10, 50, 50]
        >>> box2 = [20, 20, 60, 60]
        >>> iou = calculate_iou_corners(box1, box2)
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate individual box areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def xywh_to_corners(box):
    """
    Converts a bounding box from [x, y, w, h] format to [xmin, ymin, xmax, ymax].
    
    This conversion is necessary because different datasets use different
    bounding box formats. The corner format is used internally for IoU calculation.
    
    Args:
        box (list): Bounding box in [x, y, width, height] format
        
    Returns:
        list: Bounding box in [xmin, ymin, xmax, ymax] format
        
    Example:
        >>> box_xywh = [100, 100, 50, 80]
        >>> box_corners = xywh_to_corners(box_xywh)
        >>> print(box_corners)  # [100, 100, 150, 180]
    """
    x, y, w, h = box
    return [x, y, x + w, y + h]


def rescale_box_from_1000px(box, original_w, original_h):
    """
    Rescales a bounding box from normalized [0, 1000] space to pixel coordinates.
    
    VLM models typically output bounding boxes normalized to a 1000x1000 space.
    This function converts those coordinates to the original image dimensions.
    
    Args:
        box (list): Bounding box in [x1, y1, x2, y2] format, normalized to [0, 1000]
        original_w (int): Original image width in pixels
        original_h (int): Original image height in pixels
        
    Returns:
        list: Bounding box in pixel coordinates [xmin, ymin, xmax, ymax],
              clamped to image boundaries
              
    Example:
        >>> box_normalized = [100, 200, 500, 600]  # In 1000x1000 space
        >>> box_pixels = rescale_box_from_1000px(box_normalized, 1920, 1080)
    """
    MODEL_SPACE_DIM = 1000.0
    
    x1, y1, x2, y2 = box
    
    # Convert from [0, 1000] scale to [0, 1] scale
    x1_norm = x1 / MODEL_SPACE_DIM
    y1_norm = y1 / MODEL_SPACE_DIM
    x2_norm = x2 / MODEL_SPACE_DIM
    y2_norm = y2 / MODEL_SPACE_DIM
    
    # Scale to original image dimensions
    x1_orig = x1_norm * original_w
    y1_orig = y1_norm * original_h
    x2_orig = x2_norm * original_w
    y2_orig = y2_norm * original_h

    # Clamp coordinates to image boundaries
    rescaled_box = [
        max(0, int(x1_orig)),
        max(0, int(y1_orig)),
        min(original_w, int(x2_orig)),
        min(original_h, int(y2_orig))
    ]

    return rescaled_box

    
def convert_to_python_types(data):
    """
    Recursively converts NumPy types to native Python types.
    
    This is necessary for JSON serialization of results, as NumPy types
    (np.int64, np.float32, etc.) are not JSON-serializable.
    
    Args:
        data: Input data (can be nested lists, dicts, or scalar values)
        
    Returns:
        Data structure with all NumPy types converted to Python types
        
    Example:
        >>> data = {'value': np.float32(3.14), 'list': [np.int64(1), np.int64(2)]}
        >>> python_data = convert_to_python_types(data)
        >>> json.dumps(python_data)  # Now works without errors
    """
    if isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    if isinstance(data, dict):
        return {key: convert_to_python_types(value) for key, value in data.items()}
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    return data


class Summary(Enum):
    """
    Enumeration for metric summary types.
    
    Used with AverageMeter to specify how metrics should be summarized
    when printing results.
    """
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """
    Computes and stores running average and current value of a metric.
    
    Useful for tracking metrics like loss, accuracy, or IoU during evaluation.
    Supports distributed training with the all_reduce method.
    
    Attributes:
        name (str): Name of the metric
        val (float): Current value
        avg (float): Running average
        sum (float): Running sum
        count (int): Number of updates
        
    Args:
        name (str): Name of the metric for display
        fmt (str): Format string for printing (default: ':.4f')
        summary_type (Summary): How to summarize the metric (default: AVERAGE)
        
    Example:
        >>> iou_meter = AverageMeter('IoU', fmt=':.4f')
        >>> iou_meter.update(0.75, n=1)
        >>> iou_meter.update(0.80, n=1)
        >>> print(iou_meter.avg)  # 0.775
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """Resets all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.
        
        Args:
            val (float): New value to add
            n (int): Weight/count for this value (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """
        Synchronizes metrics across distributed processes.
        
        Uses torch.distributed to sum values across all processes,
        enabling accurate metrics in multi-GPU training/evaluation.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist() + [self.count],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        """Returns formatted string with current and average values."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """Returns summary string based on configured summary_type."""
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def encode_masks_to_rle(masks):
    """
    Encodes binary masks to Run-Length Encoding (RLE) format.
    
    RLE is a compact representation for binary masks used in COCO-style
    annotations. This function converts a list of binary masks to RLE format.
    
    Args:
        masks (list): List of binary mask arrays (numpy arrays)
        
    Returns:
        list: List of RLE-encoded masks (dictionaries with 'size' and 'counts')
        
    Example:
        >>> mask = np.zeros((480, 640), dtype=np.uint8)
        >>> mask[100:200, 100:200] = 1
        >>> rle_masks = encode_masks_to_rle([mask])
    """
    rle_list = []
    for mask in masks:
        # Ensure mask is Fortran-ordered array of uint8 for pycocotools
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_list.append(rle)
    return rle_list


def save_dataset_video(output_filename, video_path, ground_truth_tube=None, pred_tube=None):
    """
    Creates a visualization video with ground truth and predicted bounding boxes.
    
    Reads frames from the input video and overlays bounding boxes for visual
    inspection of model predictions vs. ground truth annotations.
    
    Args:
        output_filename (str): Path for the output video file
        video_path (str): Path to the input video file
        ground_truth_tube (dict, optional): Frame-indexed ground truth boxes
            Format: {frame_idx: [x, y, w, h]}
        pred_tube (dict, optional): Frame-indexed predicted boxes
            Format: {frame_idx: [xmin, ymin, xmax, ymax]}
            
    Raises:
        AssertionError: If video capture fails to open
        RuntimeError: If video writer fails to initialize
        
    Example:
        >>> gt = {0: [100, 100, 50, 50], 1: [105, 105, 50, 50]}
        >>> pred = {0: [98, 102, 148, 152], 1: [103, 107, 153, 157]}
        >>> save_dataset_video('output.mp4', 'input.mp4', gt, pred)
    """
    cap = cv2.VideoCapture(video_path)
    assert cap is not None, '  WARNING: failed to open video, capture is None'

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default FPS if not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video for writing: {output_filename}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ground truth box (green)
        if ground_truth_tube:
            gt_bbox = ground_truth_tube.get(frame_idx, [0, 0, 0, 0])
            x, y, w, h = gt_bbox
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, "Ground Truth", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Draw predicted box (red)
        if pred_tube:
            pred_box = pred_tube.get(frame_idx, [0, 0, 0, 0])
            pred_pt1, pred_pt2 = (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3])
            cv2.rectangle(frame, pred_pt1, pred_pt2, (0, 0, 255), 2)
            cv2.putText(frame, "Prediction", (pred_pt1[0], pred_pt2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1
        
    print(f'  SUCCESS: saved video file as {output_filename}')
    cap.release()
    out.release()