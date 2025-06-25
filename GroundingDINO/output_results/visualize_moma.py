import os
import json
import cv2
import numpy as np

# --------------------------
# Load JSON Annotations
# --------------------------

annotations_file = "/home/da530038/datasets_download/MOMA/anns/anns/graph_anns.json"
video_path = "/home/da530038/groudingdino/GroundingDINO/output_results/20201115222849.mp4"
output_video_path = "output_visualization.mp4"

# Load annotations
with open(annotations_file, "r") as f:
    annotations = json.load(f)

# --------------------------
# Filter Annotations for This Video
# --------------------------

trim_video_id = "20201115222849"  # Target trimmed video ID
filtered_annotations = [
    ann for ann in annotations if ann["trim_video_id"] == trim_video_id]

if not filtered_annotations:
    print(f"❌ No annotations found for video {trim_video_id}")
    exit()

# --------------------------
# Open Video File
# --------------------------

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Error: Cannot open video {video_path}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --------------------------
# Convert Frame Timestamp (Seconds) to Frame Index
# --------------------------
annotation_map = {}  # Dictionary to store annotations for each frame range

for ann in filtered_annotations:
    timestamp_sec = int(ann["frame_timestamp"])  # Convert to integer seconds
    frame_start = timestamp_sec * fps  # Convert seconds to starting frame index
    # Keep annotation visible for 1 second
    frame_end = (timestamp_sec + 1) * fps

    # Store annotation for that time range
    annotation_map[(frame_start, frame_end)] = ann

# --------------------------
# Process Each Frame
# --------------------------

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Find annotations for this frame range
    for (start_frame, end_frame), ann in annotation_map.items():
        if start_frame <= frame_count < end_frame:
            # Draw actor bounding boxes (Green)
            for actor in ann["annotation"]["actors"]:
                bbox = actor["bbox"]
                x1, y1 = int(bbox["topLeft"]["x"]), int(bbox["topLeft"]["y"])
                x2, y2 = int(bbox["bottomRight"]["x"]), int(
                    bbox["bottomRight"]["y"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Actor: {actor['class']}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw object bounding boxes (Blue)
            for obj in ann["annotation"]["objects"]:
                bbox = obj["bbox"]
                x1, y1 = int(bbox["topLeft"]["x"]), int(bbox["topLeft"]["y"])
                x2, y2 = int(bbox["bottomRight"]["x"]), int(
                    bbox["bottomRight"]["y"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Object: {obj['class']}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw atomic actions (Red text)
            for action in ann["annotation"]["atomic_actions"]:
                action_class = action["class"]
                cv2.putText(frame, f"Action: {action_class}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)
    frame_count += 1

# Release resources
cap.release()
out.release()
print(f"✅ Visualization saved: {output_video_path}")
