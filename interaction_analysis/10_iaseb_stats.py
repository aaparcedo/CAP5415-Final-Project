# %%
import json
from constants import DATASET_PATHS
import os
from pathlib import Path
import cv2
from tqdm import tqdm
# %%
path = "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_metadata.json"
data = json.load(open(path, 'r'))
# %%
captions = [entry["caption"] for entry in data]
caption_sizes = [len(caption.split(" ")) for caption in captions]

unique_video_paths = len(set([entry["video_path"] for entry in data]))
average_text_length = sum(caption_sizes) / len(captions)
average_num_frames = sum([int(entry["tube_end_frame"]) - int(entry["tube_start_frame"]) for entry in data]) / len(captions)
average_num_words = sum(caption_sizes) / len(captions)


print('=' * 100)
print(f'Total number of videos: {unique_video_paths}')
print(f'Unique video:caption:box/mask triplets: {len(data)}')
print(f'Average text length: {average_text_length:.2f}')
print(f'Average number of frames (trimmed), i.e., average tube size: {average_num_frames:.2f}')
print(f'Average number of words in caption: {average_num_words}')

print('=' * 100)

# %%

print("Creating histogram plot...")
import seaborn as sns
import matplotlib.pyplot as plt

    # Set the theme to match your reference
sns.set_theme(style="darkgrid")
plt.figure(figsize=(7, 5)) # Smaller figure size, good for histograms

# Use the same color as your other plot
plot_color = "#8da2cc"

    # Create the histogram
sns.histplot(
    data=caption_sizes,
    bins=20,                 # 20 bins, just like your reference
    # binrange=(0.0, 1.0),     # Lock the range from 0 to 1
    color=plot_color
)

    # 3. Set Ticks and Labels (to match reference)
ax = plt.gca() # Get the current axis
    
    # # Set the x-axis ticks
    # ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    
    # Set labels
ax.set_xlabel("Number of words in input query")
    # ax.set_ylabel("Count")
    # ax.set_title("") # No main title, as per your reference
    
plt.tight_layout()

    # 4. Save the figure
    
plt.savefig('caption_length_distribution.svg', format='svg', bbox_inches='tight')
plt.show()
# %%


# %%

widths = []
heights = []

for sample in tqdm(data):
    dataset = sample["dataset"]

    video_path = Path(sample["video_path"])

    if not video_path.is_absolute():
        
        video_path = Path(DATASET_PATHS[dataset]['video']) / video_path

    if dataset == 'mevis' or dataset == 'rvos': # mevis and rvos; not a video
        image = cv2.imread(os.path.join(video_path, os.listdir(video_path)[0])) # read the first image
        height = image.shape[0]
        width = image.shape[1]
    else:
        cap = cv2.VideoCapture(video_path)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    widths.append(width)
    heights.append(height)

    # duration_seconds = 0
    # if fps > 0:
    #     duration_seconds = total_frames / fps
    # else:
    #     print("Warning: FPS is zero, cannot calculate duration.")

    # if not dataset == 'hcstvg1' or not dataset=='hcstvg2':
    #     sample["frame_count"] = total_frames
    #     sample["width"] = frame_width
    #     sample["height"] = frame_height
    # sample["duration_seconds"] = duration_seconds
    # sample["fps"] = fps
    cap.release()

average_width = sum(widths) / len(data)
average_height = sum(heights) / len(data)

print(f'average width: {average_width}')
print(f'average height: {average_height}')
# average width: 865.8269613259669 -> 866 
# average height: 544.759926335175 -> 545
# %%