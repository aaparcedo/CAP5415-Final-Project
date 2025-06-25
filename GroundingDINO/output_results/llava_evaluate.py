# import json
# import numpy as np
# import pandas as pd


# def iou(box1, box2):
#     """Calculate Intersection over Union (IoU) between two bounding boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = box1_area + box2_area - intersection

#     return intersection / union if union > 0 else 0


# def evaluate_predictions(json_path):
#     """Evaluate predicted bounding boxes using frame-wise IoU averaging."""
#     with open(json_path, "r") as file:
#         data = json.load(file)

#     all_video_ious = []
#     vIoU_03 = 0
#     vIoU_05 = 0
#     total_videos = 0

#     for sample in data:
#         # List of GT boxes for each frame
#         gt_boxes = sample["ground_truth_boxes"]
#         # List of predicted boxes for each frame
#         pred_boxes = sample["predicted_boxes"]

#         if len(gt_boxes) == 0 or len(pred_boxes) == 0:
#             continue  # Skip videos with missing data

#         frame_ious = [iou(pred, gt) for pred, gt in zip(pred_boxes, gt_boxes)]

#         mean_frame_iou = np.mean(frame_ious) if frame_ious else 0.0
#         all_video_ious.append(mean_frame_iou)

#         if mean_frame_iou >= 0.3:
#             vIoU_03 += 1
#         if mean_frame_iou >= 0.5:
#             vIoU_05 += 1
#         total_videos += 1

#     mean_vIoU = np.mean(all_video_ious) if all_video_ious else 0
#     vIoU_03_perc = (vIoU_03 / total_videos) * 100 if total_videos else 0
#     vIoU_05_perc = (vIoU_05 / total_videos) * 100 if total_videos else 0

#     return {
#         "m_vIoU": round(mean_vIoU * 100, 2),  # Convert to percentage
#         "vIoU@0.3": round(vIoU_03_perc, 2),
#         "vIoU@0.5": round(vIoU_05_perc, 2)
#     }


# def main():
#     results = evaluate_predictions("predictions_2.json")
#     df = pd.DataFrame([results], index=["GroudingDINO"])
#     print(df)


# if __name__ == "__main__":
#     main()


import json
import numpy as np
import pandas as pd


def iou(box1, box2):
    """
    Computes IoU between two bounding boxes in [x, y, w, h] format.

    Args:
        box1 (list): First box in [x, y, w, h] format.
        box2 (list): Second box in [x, y, w, h] format.

    Returns:
        float: IoU score.
    """
    # Convert [x, y, w, h] → [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    # x2 = x1 + w, y2 = y1 + h
    box1_x2, box1_y2 = box1_x1 + box1[2], box1_y1 + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2_x1 + box2[2], box2_y1 + box2[3]

    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate areas
    area1 = box1[2] * box1[3]  # width * height
    area2 = box2[2] * box2[3]

    # Compute IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def evaluate_predictions(json_path):
    """Evaluate predicted bounding boxes using 15-frame jump IoU averaging."""
    with open(json_path, "r") as file:
        data = json.load(file)

    all_video_ious = []
    vIoU_03 = 0
    vIoU_05 = 0
    total_videos = 0

    for sample in data:
        gt_boxes = sample["ground_truth_boxes"]
        pred_boxes = sample["predicted_boxes"]
        # if (len(gt_boxes) < len(pred_boxes)):
        #     print(f'Mismatch:{len(gt_boxes)} - {len(pred_boxes)}')
        gt_boxes = gt_boxes[::15]
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue  # Skip videos with missing data

        # Evaluate with 15-frame jump
        frame_ious = [
            iou(pred_boxes[i], gt_boxes[i])
            for i in range(0, min(len(gt_boxes), len(pred_boxes)))
        ]

        mean_frame_iou = np.mean(frame_ious) if frame_ious else 0.0
        all_video_ious.append(mean_frame_iou)

        if mean_frame_iou >= 0.3:
            vIoU_03 += 1
        if mean_frame_iou >= 0.5:
            vIoU_05 += 1
        total_videos += 1

    mean_vIoU = np.mean(all_video_ious) if all_video_ious else 0
    vIoU_03_perc = (vIoU_03 / total_videos) * 100 if total_videos else 0
    vIoU_05_perc = (vIoU_05 / total_videos) * 100 if total_videos else 0

    return {
        "m_vIoU": round(mean_vIoU * 100, 2),  # Convert to percentage
        "vIoU@0.3": round(vIoU_03_perc, 2),
        "vIoU@0.5": round(vIoU_05_perc, 2)
    }


def main():
    results = evaluate_predictions(
        "/home/da530038/llava-grouding/LLaVA-Grounding/results/referral_hcstvg2_iou_llava_predictions.json")
    df = pd.DataFrame([results], index=["LLaVa Grounding"])
    print(df)


if __name__ == "__main__":
    main()
