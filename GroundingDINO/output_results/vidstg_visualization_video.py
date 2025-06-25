import json
import os
import cv2
import numpy as np

# --------------------------
# Load Predictions JSON
# --------------------------

predictions_file = "vidstg_predictions.json"  # Update with your file
video_dir = "/home/c3-0/datasets/stvg/vidstg/video/validation"
output_dir = "output_videos_vidstg"

os.makedirs(output_dir, exist_ok=True)

# Load predictions JSON
with open(predictions_file, "r") as f:
    predictions = json.load(f)

# --------------------------
# Process Each Video Entry
# --------------------------

for entry in predictions:
    video_path = entry["video_path"]
    caption = entry["caption"]
    width = entry["width"]
    height = entry["height"]

    # Dict {frame_idx: [x, y, w, h]}
    ground_truth_boxes = entry["ground_truth_boxes"]
    predicted_boxes = entry["predicted_boxes"]  # List of [x, y, w, h]

    # Extract just the filename, ignoring directories
    video_filename = os.path.basename(video_path).replace(".mp4", "")
    print(video_filename)
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))

    frame_count = 0
    predicted_index = 0  # Index for iterating through predicted boxes

    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break

        frame_key = str(frame_idx)  # Convert frame index to string

        # Draw ground truth box (green) if available for the current frame
        if frame_key in ground_truth_boxes:
            gt_box = ground_truth_boxes[frame_key]  # [x, y, w, h]
            x, y, w, h = map(int, gt_box)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)  # Green box

            # Draw corresponding predicted box (red), assuming sequential order
            # if predicted_index < len(predicted_boxes):
            #     pred_box = predicted_boxes[predicted_index]  # [x, y, w, h]
            #     x, y, w, h = map(int, pred_box)
            #     cv2.rectangle(frame, (x, y), (x + w, y + h),
            #                   (0, 0, 255), 2)  # Red box
            #     predicted_index += 1  # Move to the next predicted box

        # Put caption text on the first frame
        if frame_count == 0:
            cv2.putText(frame, caption, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write frame to output video
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Visualization saved: {output_video_path}")
