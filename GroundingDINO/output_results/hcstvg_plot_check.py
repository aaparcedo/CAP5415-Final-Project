import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the JSON data
with open('predictions_2.json', 'r') as file:
    data = json.load(file)

print("Successfully loaded JSON data.")


def plot_boxes_on_xy(video_data, first_frame_only=False):
    """
    Plots bounding boxes on an empty X-Y axis without using video frames.

    Args:
        video_data (dict): Video data containing bounding boxes.
        first_frame_only (bool): If True, only plots the first frame's boxes.
    """

    video_path = video_data['video_path']
    caption = video_data.get('caption', 'No caption available')
    width = video_data['width']
    height = video_data['height']
    ground_truth_boxes = video_data['ground_truth_boxes']
    predicted_boxes = video_data['predicted_boxes']

    # If no bounding boxes exist, skip this video
    if not ground_truth_boxes and not predicted_boxes:
        print(f"Skipping {video_path}: No bounding boxes found.")
        return

    # Select only the first bounding box if required
    if first_frame_only:
        ground_truth_boxes = [ground_truth_boxes[0]
                              ] if ground_truth_boxes else []
        predicted_boxes = [predicted_boxes[0]] if predicted_boxes else []

    # Create an empty X-Y axis plot
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.set_title(f"Video: {video_path}\nCaption: {caption}", fontsize=10)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip Y-axis so (0,0) is at the top-left
    ax.set_xlabel("X-axis (width)")
    ax.set_ylabel("Y-axis (height)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Draw ground truth boxes (Green)
    for box in ground_truth_boxes:
        if len(box) != 4:
            print(f"Skipping malformed ground truth box: {box}")
            continue
        x, y, w, h = box  # Using [x, y, w, h]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', label="Ground Truth"
        )
        ax.add_patch(rect)

    # Draw predicted boxes (Red)
    for box in predicted_boxes:
        if len(box) != 4:
            print(f"Skipping malformed predicted box: {box}")
            continue
        x, y, w, h = box  # Using [x, y, w, h]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', label="Predicted"
        )
        ax.add_patch(rect)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Save the visualization
    suffix = "_first_frame" if first_frame_only else "_full"
    output_filename = f"bounding_box{suffix}.png"
    plt.savefig(output_filename)
    plt.close(fig)

    print(f"Saved bounding box plot for {video_path} as {output_filename}")


# --------------------------
# Run the Visualization
# --------------------------

for video in data:
    plot_boxes_on_xy(video, first_frame_only=True)  # First frame only
    plot_boxes_on_xy(video, first_frame_only=False)  # All frames

print("Visualization complete.")
