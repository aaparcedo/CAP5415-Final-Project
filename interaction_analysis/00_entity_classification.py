import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# --- Configuration ---
try:
    client = OpenAI()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found.")
except Exception as e:
    print(f"Error: Could not initialize OpenAI client. {e}")
    exit()

SAMPLE_SIZE = 99999 # You can increase this for the final run

# --- File and Model Settings ---
INPUT_JSON = '/home/aparcedo/IASEB/clustering/hcstvg1_hcstvg2_vidvrd_vidstg_captions_with_dataset.json'
OUTPUT_CSV = f'hcstvg12_vidvrdstg_gpt4omini_entity_class_{SAMPLE_SIZE}subset_v1.csv'
OUTPUT_CSV = f'hcstvg12_vidvrdstg_gpt4omini_entity_class_v1.csv'
MODEL_NAME = "gpt-4o-mini"


HIERARCHY = """
1.0 Human-Human
    1.1 Cooperative (e.g., helping, exchanging)
    1.2 Competitive (e.g., fighting, sports)
    1.3 Affective (e.g., hugging, arguing, kissing)
    1.4 Proximity (e.g., person A is behind person B, standing near, away)
    1.5 Observation (e.g., watching, looking at)
    1.6 Spatial (e.g., human/object A is larger/smaller than human/object B)
2.0 Human-Object
    2.1 Active Manipulation (e.g., opening, cutting, riding bicycle, holding, carrying)
    2.2 Proximity (e.g., person A is behind object B, standing near, away, pass)
    2.3 Passive (e.g., sitting, wearing)
    2.4 Spatial (e.g., human/object A is larger/smaller than human/object B)
3.0 Human-Animal
    3.1 Direct Interaction (e.g., petting, feeding, touching)
    3.2 Observation (e.g., watching)
    3.3 Proximity (e.g., standing near)
    3.4 Spatial (e.g., human/animal A is larger/smaller than human/animal B)
4.0 Animal-Animal
    4.1 Proximity (e.g., playing, standing near, to the right/left of)
    4.2 Antagonistic (e.g., fighting, hunting)
    4.3 Observation (e.g., watching, looking at)
    4.4 Spatial (e.g., animal A is larger/smaller than animal B)
5.0 Animal-Object
    5.1 Interaction (e.g., playing with toy, building nest)
    5.2 Proximity (e.g., standing near, away)
    5.3 Spatial (e.g., animal/object A is larger/smaller than animal/object B)
6.0 Object-Object
    6.1 Spatial/Movement (e.g., car moving near another car)
    6.2 Proximity (e.g., object A is beneath/above object B, object C is away/close from/to object D)
    6.3 Spatial (e.g., object A is larger/smaller than object B)
7.0 Human-Self
    7.1 Self-interaction (e.g., cover mouth, raise hand)
8.0 No Interaction
    8.1 A single agent or object acting in isolation.
"""

FEW_SHOT_EXAMPLES = """
Here are examples of correct classifications for tricky cases. Focus on the primary interaction.
- Caption: "The sitting man turns his head to look at the standing man, and leans against the wall."
- Correct Classification: 1.5 Observation

- Caption: "The man bends over and covers his mouth."
- Correct Classification: 7.1 Self-interaction

- Caption: "A woman wearing a red hat sits on the bench."
- Correct Classification: 2.3 Passive

- Caption: "there is a yellow chair beneath an adult in a gym."
- Correct Classification: 2.2 Proximity

- Caption: "there is a laptop in front of a child in a room."
- Correct Classification: 2.2 Proximity

- Caption: "The person touches the dog."
- Correct Classification: 3.1 Direct Interaction

- Caption: "The larger of the two zebras"
- Correct Classification: 4.4 Spatial

- Caption: "The elephant plays with the ball."
- Correct Classification: 5.1 Interaction

- Caption: "The whale swims to the right of the watercraft"
- Correct Classification: 5.2 Proximity

- Caption: "there is a white vase above a brown table."
- Correct Classification: 6.2 Proximity

- Caption: "The man with glasses gets up from the sofa and walks to the door"
- Correct Classification: 8.1 No Interaction

- Caption: "The woman in black clothes turns around."
- Correct Classification: 8.1 No Interaction

- Caption: "a child in red clothes is away from a red toy"
- Correct Classification: 2.2 Proximity
"""

def classify(caption: str) -> str:
    """
    Uses the OpenAI API with enhanced few-shot examples to classify a caption.
    """
    system_prompt = f"""
    You are a precise video scene classifier. Your task is to classify a caption
    Your task is to classify a caption into the most specific category from the provided hierarchy. 
    Identity the type of interaction in the scene. 

    Hierarchy:
    {HIERARCHY}

    Use these examples to guide your reasoning:
    {FEW_SHOT_EXAMPLES}
    """
    
    user_prompt = f"\n- Caption: \"{caption}\"\n- Correct Classification: "

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API Error: {e}"

def main():
    print(f"--- Starting Final Classification Run using {MODEL_NAME} ---")


    df = pd.read_json(INPUT_JSON)
    

    if len(df) > SAMPLE_SIZE:
        # df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
        df_sample = df.sample(n=SAMPLE_SIZE).copy()
    else:
        df_sample = df.copy()

    results = []
    for caption in tqdm(df_sample['caption'], desc="Classifying Captions"):
        category = classify(caption)
        results.append(category)
    
    df_sample['predicted_category_final'] = results
    
    print(f"\nSaving classified data to '{OUTPUT_CSV}'...")
    df_sample.to_csv(OUTPUT_CSV, index=False)
    

if __name__ == "__main__":
    main()