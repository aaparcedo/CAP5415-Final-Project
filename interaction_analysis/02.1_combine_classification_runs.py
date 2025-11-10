import pandas as pd
import json
FILEPATH1 =  "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv"
FILEPATH2 = "/home/aparcedo/IASEB/interaction_analysis/mevis_rvos_gpt4omini_entity_class_v1.json"


df = pd.read_csv(FILEPATH1)
# json_str = df.to_json(orient='records', lines=True)
vg12_vrd_stg_data = df.to_dict(orient="records")

mevisrvos_data = json.load(open(FILEPATH2, 'r')) # loads a list

entity_combined = vg12_vrd_stg_data + mevisrvos_data
# output_path = "vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_v1.json"
# with open(output_path, 'w') as f:
#     json.dump(combined, f, indent=4)
# %%
import json
import pandas as pd

FILEPATH1 =  "/home/aparcedo/IASEB/interaction_analysis/hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv"
FILEPATH2 = "/home/aparcedo/IASEB/interaction_analysis/mevis_rvos_gpt4omini_st_class_v1.json"

df = pd.read_csv(FILEPATH1)
# json_str = df.to_json(orient='records', lines=True)
vg12_vrd_stg_data = df.to_dict(orient="records")

mevisrvos_data = json.load(open(FILEPATH2, 'r')) # loads a list

st_combined = vg12_vrd_stg_data + mevisrvos_data
# output_path = "vg12_vrd_stg_mevis_rvos_gpt4omini_st_class_v1.json"
# with open(output_path, 'w') as f:
#     json.dump(combined, f, indent=4)
# %%
# %%
