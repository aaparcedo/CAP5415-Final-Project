# import json
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Read the JSON data from the file
# with open('vidstg_predictions.json', 'r') as file:
#     data = json.load(file)

# print("Successfully loaded JSON data.")

# def plot_boxes(video_data):
#     video_path = video_data.get('video_path', 'Unknown')
#     width = video_data.get('width', 1280)  # Default width if missing
#     height = video_data.get('height', 720)  # Default height if missing
#     ground_truth_boxes = video_data.get('ground_truth_boxes', {})
#     predicted_boxes = video_data.get('predicted_boxes', [])

#     # Ensure ground truth boxes are stored as a dictionary
#     if isinstance(ground_truth_boxes, dict):
#         ground_truth_boxes = list(ground_truth_boxes.values())

#     # Check if bounding boxes are empty
#     if not ground_truth_boxes and not predicted_boxes:
#         print(f"Skipping {video_path}: No bounding boxes found.")
#         return

#     # Create a plot for visualization
#     fig, ax = plt.subplots(1, figsize=(10, 8))
#     ax.set_title(f"Visualization for {video_path}")
#     ax.set_xlim(0, width)
#     ax.set_ylim(height, 0)
#     ax.set_xlabel('Width')
#     ax.set_ylabel('Height')

#     # Draw ground truth boxes (green)
#     for box in ground_truth_boxes:
#         if len(box) != 4:
#             print(f"Skipping malformed ground truth box: {box}")
#             continue
#         x, y, w, h = box  # Adjust this if necessary
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth')
#         ax.add_patch(rect)

#     # Draw predicted boxes (red)
#     for box in predicted_boxes:
#         if len(box) != 4:
#             print(f"Skipping malformed predicted box: {box}")
#             continue
#         x, y, w, h = box  # Adjust this if necessary
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Predicted')
#         ax.add_patch(rect)

#     # Avoid duplicate legends
#     handles, labels = ax.get_legend_handles_labels()
#     unique_labels = dict(zip(labels, handles))
#     ax.legend(unique_labels.values(), unique_labels.keys())

#     print(f"Saving plot for {video_path} ...")
#     plt.savefig(f"visualization_{video_path.replace('/', '_').replace('.mp4', '')}.png")
#     plt.close(fig)

# # Visualize each video frame
# for video in data:
#     plot_boxes(video)

import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Read the JSON data from the file
with open('vidstg_predictions.json', 'r') as file:
    data = json.load(file)

print("Successfully loaded JSON data.")


def plot_first_frame_box(video_data):
    video_path = video_data.get('video_path', 'Unknown')
    caption = video_data.get('caption', 'No caption available')
    width = video_data.get('width', 1280)  # Default width if missing
    height = video_data.get('height', 720)  # Default height if missing
    ground_truth_boxes = video_data.get('ground_truth_boxes', {})
    predicted_boxes = video_data.get('predicted_boxes', [])

    # Ensure ground truth boxes are stored as a dictionary
    if isinstance(ground_truth_boxes, dict):
        ground_truth_boxes = list(ground_truth_boxes.values())

    # Extract only the first bounding box (if available)
    first_gt_box = ground_truth_boxes[0] if ground_truth_boxes else None
    first_pred_box = predicted_boxes[0] if predicted_boxes else None

    # Check if both are missing
    if first_gt_box is None and first_pred_box is None:
        print(f"Skipping {video_path}: No bounding boxes found.")
        return

    # Capture the first frame from the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Skipping {video_path}: Unable to read the first frame.")
        return

    # Convert frame from BGR (OpenCV) to RGB (Matplotlib)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a figure
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(frame)
    ax.set_title(f"Caption: {caption}", fontsize=12, color='black', pad=10)

    # Draw only the first ground truth box (Green)
    if first_gt_box and len(first_gt_box) == 4:
        x, y, w, h = first_gt_box
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', label="Ground Truth")
        ax.add_patch(rect)

    # Draw only the first predicted box (Red)
    if first_pred_box and len(first_pred_box) == 4:
        x, y, w, h = first_pred_box
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', label="Predicted")
        ax.add_patch(rect)

    # Add a legend
    ax.legend()

    # Save the visualization
    output_filename = f"visualization_{video_path.split('/')[-1].replace('.mp4', '')}.png"
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Saved plot for {video_path} as {output_filename}")


# Process only the first video
for video in data:
    plot_first_frame_box(video)
    break  # Stops after processing one video
