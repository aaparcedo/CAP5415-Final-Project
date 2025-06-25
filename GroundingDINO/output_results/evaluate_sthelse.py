import json
import numpy as np
import pandas as pd


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def evaluate_somethingelse_predictions_from_scratch(json_path, frame_step=15):
    with open(json_path, "r") as f:
        data = json.load(f)

    all_video_ious = []
    vIoU_03, vIoU_05 = 0, 0

    for entry in data:
        gt_dict = entry.get("ground_truth_boxes", {})
        pred_boxes = entry.get("predicted_boxes", [])

        if not gt_dict or not pred_boxes:
            continue

        # Infer frame IDs from ground truth
        sorted_gt_frames = sorted([int(f) for f in gt_dict.keys()])
        if not sorted_gt_frames:
            continue

        start_frame = sorted_gt_frames[0]
        frame_ids = [start_frame + i * frame_step for i in range(len(pred_boxes))]

        frame_ious = []
        for pred_box, fid in zip(pred_boxes, frame_ids):
            fid_str = str(fid)
            gt_box = gt_dict.get(fid_str)
            if gt_box is not None:
                iou = calculate_iou(pred_box, gt_box)
                frame_ious.append(iou)

        if frame_ious:
            mv_iou = np.mean(frame_ious)
            all_video_ious.append(mv_iou)

            if mv_iou >= 0.3:
                vIoU_03 += 1
            if mv_iou >= 0.5:
                vIoU_05 += 1

    num_videos = len(all_video_ious)
    return {
        "m_vIoU": round(np.mean(all_video_ious) * 100, 2) if num_videos else 0,
        "vIoU@0.3": round((vIoU_03 / num_videos) * 100, 2) if num_videos else 0,
        "vIoU@0.5": round((vIoU_05 / num_videos) * 100, 2) if num_videos else 0,
        "Num samples": num_videos
    }


def main():
    json_path = "/home/da530038/groudingdino/GroundingDINO/output_results/somethingelse_predictions.json"
    results = evaluate_somethingelse_predictions_from_scratch(json_path, frame_step=15)
    df = pd.DataFrame([results], index=["GroundingDINO"])
    print(df)


if __name__ == "__main__":
    main()
