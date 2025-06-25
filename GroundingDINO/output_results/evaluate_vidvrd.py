import json
import numpy as np
import pandas as pd


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes: [xmin, ymin, xmax, ymax]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def evaluate_vidvrd_predictions_from_scratch(json_path, frame_step=15):
    with open(json_path, "r") as f:
        data = json.load(f)

    all_video_ious = []
    vIoU_03, vIoU_05 = 0, 0

    for sample in data:
        gt_dict = sample.get("ground_truth_boxes", {})
        pred_boxes = sample.get("predicted_boxes", [])
        frame_ids = sample.get("frame_ids", [])

        if not gt_dict or not pred_boxes or not frame_ids:
            continue

        frame_ious = []
        for pred_box, frame_id in zip(pred_boxes, frame_ids):
            if frame_id % frame_step != 0:
                continue
            frame_key = str(frame_id)
            if frame_key in gt_dict:
                gt_box = gt_dict[frame_key]
                iou_val = calculate_iou(pred_box, gt_box)
                frame_ious.append(iou_val)

        if frame_ious:
            video_mv_iou = np.mean(frame_ious)
            all_video_ious.append(video_mv_iou)

            if video_mv_iou >= 0.3:
                vIoU_03 += 1
            if video_mv_iou >= 0.5:
                vIoU_05 += 1

    num_videos = len(all_video_ious)
    m_vIoU = round(np.mean(all_video_ious) * 100, 2) if num_videos else 0
    vIoU_03_score = round((vIoU_03 / num_videos) * 100, 2) if num_videos else 0
    vIoU_05_score = round((vIoU_05 / num_videos) * 100, 2) if num_videos else 0

    return {
        "m_vIoU": m_vIoU,
        "vIoU@0.3": vIoU_03_score,
        "vIoU@0.5": vIoU_05_score,
        "Num samples": num_videos
    }


def main():
    result_path = "/home/da530038/groudingdino/GroundingDINO/output_results/vidvrd_predictions_referral_full.json"
    results = evaluate_vidvrd_predictions_from_scratch(result_path)
    df = pd.DataFrame([results], index=["GroundingDINO"])
    print(df)


if __name__ == "__main__":
    main()
