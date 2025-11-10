from pycocotools import mask as mask_util
from enum import Enum
import numpy as np
import torch
import torch.distributed as dist
import cv2

def calculate_iou_corners(box1, box2):
    """
    Computes IoU for boxes in [xmin, ymin, xmax, ymax] format.
    Used for VidVRD and Something-Else datasets.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def xywh_to_corners(box):
    """Converts a bounding box from [x, y, w, h] to [xmin, ymin, xmax, ymax]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]

def rescale_box_from_1000px(box, original_w, original_h):
    """
    Rescales a bounding box from the model's normalized [0, 1000] space
    to the original image's pixel dimensions.
    """
    # The model's raw output space is normalized to 1000x1000
    MODEL_SPACE_DIM = 1000.0
    
    x1, y1, x2, y2 = box
    
    # Convert from [0, 1000] scale to [0, 1] scale
    x1_norm = x1 / MODEL_SPACE_DIM
    y1_norm = y1 / MODEL_SPACE_DIM
    x2_norm = x2 / MODEL_SPACE_DIM
    y2_norm = y2 / MODEL_SPACE_DIM
    
    # Scale the normalized coordinates to the original image dimensions
    x1_orig = x1_norm * original_w
    y1_orig = y1_norm * original_h
    x2_orig = x2_norm * original_w
    y2_orig = y2_norm * original_h

    # Clamp coordinates to ensure they are within the original frame's bounds
    rescaled_box = [
        max(0, int(x1_orig)),
        max(0, int(y1_orig)),
        min(original_w, int(x2_orig)),
        min(original_h, int(y2_orig))
    ]

    return rescaled_box
    
def convert_to_python_types(data):
    """Recursively converts NumPy types to native Python types in a nested structure."""
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
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
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
    rle_list = []
    for mask in masks:
        # NOTE: Ensure the mask is a Fortran-ordered array of uint8 for pycocotools
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        rle_list.append(rle)
    return rle_list


def save_dataset_video(output_filename, video_path, ground_truth_tube=None, pred_tube=None):

    cap = cv2.VideoCapture(video_path)
    assert cap is not None, '  WARNING: failed to open video, capture is None'

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video for writing: {output_filename}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if ground_truth_tube:
            gt_bbox = ground_truth_tube.get(frame_idx, [0, 0, 0, 0])
            x, y, w, h = gt_bbox
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2) # draw green box for ground truth
            cv2.putText(frame, "Ground Truth", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            
        if pred_tube:
            pred_box = pred_tube.get(frame_idx, [0, 0, 0, 0]) # default box if no prediction
            pred_pt1, pred_pt2 = (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3])
            cv2.rectangle(frame, pred_pt1, pred_pt2, (0, 0, 255), 2) # draw red box for prediction
            cv2.putText(frame, "Prediction", (pred_pt1[0], pred_pt2[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(frame)
        
        frame_idx += 1
    print(f'  SUCCESS: saved video file as {output_filename}')
    cap.release()
    out.release()