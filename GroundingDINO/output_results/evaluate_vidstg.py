import json
import numpy as np
import pandas as pd


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])  # Convert width to x2
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])  # Convert height to y2

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def evaluate_predictions(json_path):
    """Evaluate bounding boxes using 15-frame jump based on structured GT dictionary and predicted list."""
    with open(json_path, "r") as file:
        data = json.load(file)

    all_video_ious = []
    vIoU_03 = 0
    vIoU_05 = 0
    total_videos = 0

    for sample in data:
        # dict with frame indices as string
        gt_dict = sample["ground_truth_boxes"]
        pred_boxes = sample["predicted_boxes"]  # list of boxes by frame

        if not gt_dict or not pred_boxes:
            continue  # Skip videos with missing data

        key_lists = list(gt_dict.keys())
        st_frame = int(key_lists[0])
        end_frame = int(key_lists[-1])
        frame_ious = []
        for frame_idx, pred_box in enumerate(pred_boxes):
            if frame_idx % 15 != 0:
                continue  # Apply 15-frame jump
            real_frame = st_frame + frame_idx
            if (real_frame > end_frame):
                break
            frame_key = str(real_frame)
            if frame_key in gt_dict:
                gt_box = gt_dict[frame_key]
                frame_ious.append(iou(pred_box, gt_box))
            else:
                print('err')

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
        "m_vIoU": round(mean_vIoU * 100, 2),
        "vIoU@0.3": round(vIoU_03_perc, 2),
        "vIoU@0.5": round(vIoU_05_perc, 2)
    }


def main():
    # results = evaluate_predictions("/home/da530038/groudingdino/GroundingDINO/output_results/vidstg_predictions_full_swinB.json")
    results = evaluate_predictions("/home/da530038/groudingdino/GroundingDINO/output_results/vidvrd_predictions_referral_full.json")
    df = pd.DataFrame([results], index=["GroundingDINO"])
    print(df)


if __name__ == "__main__":
    main()
