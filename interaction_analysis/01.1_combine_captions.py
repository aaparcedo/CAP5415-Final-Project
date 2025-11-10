# %%
import json

FILE1_PATH = '/home/aparcedo/IASEB/interaction_analysis/mevis_rvos_captions.json'
FILE2_PATH = '/home/aparcedo/IASEB/clustering/hcstvg1_hcstvg2_vidvrd_vidstg_captions_with_dataset.json'

mevisrvos = json.load(open(FILE1_PATH, 'r'))

stvg12vrdstg = json.load(open(FILE2_PATH, 'r'))

combined = stvg12vrdstg + mevisrvos

output_path = "vg12_vrd_stg_mevis_rvos_captions.json"
with open(output_path, 'w') as f:
    json.dump(combined, f, indent=4)
# %%
