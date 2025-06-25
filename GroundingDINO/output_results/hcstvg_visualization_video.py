# import json

# # --------------------------
# # Load JSON Files
# # --------------------------

# # Paths to JSON files
# predictions_file = "predictions_1_one.json"
# test_proc_file = "/home/c3-0/datasets/stvg/hcstvg1/test_proc.json"
# # predictions_file = "predictions_2_one.json"
# # test_proc_file = "/home/c3-0/datasets/stvg/hcstvg2/annotations/HCVG_val_proc.json"

# # Load predictions JSON
# with open(predictions_file, "r") as f:
#     predictions = json.load(f)

# # Load test_proc JSON
# with open(test_proc_file, "r") as f:
#     test_proc_data = json.load(f)

# # Create a dictionary mapping video_path to tube_start_frame and tube_end_frame
# video_info_map = {entry["video_path"]: (
#     entry["tube_start_frame"], entry["tube_end_frame"]) for entry in test_proc_data}

# # --------------------------
# # Update Predictions JSON
# # --------------------------

# for entry in predictions:
#     video_path = entry["video_path"]

#     if video_path in video_info_map:
#         entry["tube_start_frame"], entry["tube_end_frame"] = video_info_map[video_path]
#     else:
#         print(f"Warning: No matching video found for {video_path}")

# # --------------------------
# # Save Updated JSON
# # --------------------------

# updated_file = "predictions_1_one_updated.json"
# with open(updated_file, "w") as f:
#     json.dump(predictions, f, indent=4)

# print(f"Updated predictions saved to {updated_file}")


import json
import os
import cv2
import numpy as np

# --------------------------
# Load Predictions JSON
# --------------------------

# predictions_file = "predictions_1_one_updated.json"
# video_dir = "/home/c3-0/datasets/stvg/hcstvg1/v1/video"
# output_dir = "output_videos"
predictions_file = "predictions_2_one_updated.json"
video_dir = "/home/c3-0/datasets/stvg/hcstvg2/videos"
output_dir = "output_videos"

os.makedirs(output_dir, exist_ok=True)

# Load predictions JSON
with open(predictions_file, "r") as f:
    predictions = json.load(f)

# --------------------------
# Process Each Video Entry
# --------------------------

for entry in predictions:
    video_path = os.path.join(video_dir, entry["video_path"])
    tube_start_frame = entry["tube_start_frame"]
    tube_end_frame = entry["tube_end_frame"]
    ground_truth_boxes = entry["ground_truth_boxes"]
    predicted_boxes = entry["predicted_boxes"]

    # Extract just the filename, ignoring directories
    video_filename = os.path.basename(entry["video_path"]).replace(".mp4", "")

    # Construct the output path
    output_video_path = os.path.join(
        output_dir, f"{video_filename}_visualized.mp4")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        continue

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count < tube_start_frame:
            frame_count += 1
            continue

        if frame_count > tube_end_frame:
            break

        # Draw ground truth box (green)
        if frame_count - tube_start_frame < len(ground_truth_boxes):
            gt_box = ground_truth_boxes[frame_count - tube_start_frame]
            cv2.rectangle(frame, (gt_box[0], gt_box[1]),
                          (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]),
                          (0, 255, 0), 2)  # Green box

        # Draw predicted box (red)
        if frame_count - tube_start_frame < len(predicted_boxes):
            pred_box = predicted_boxes[frame_count - tube_start_frame]
            cv2.rectangle(frame, (int(pred_box[0]), int(pred_box[1])),
                          (int(pred_box[0] + pred_box[2]),
                           int(pred_box[1] + pred_box[3])),
                          (0, 0, 255), 2)  # Red box

        # Write frame to output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Visualization saved: {output_video_path}")
