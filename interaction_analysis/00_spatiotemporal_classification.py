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

# --- File and Model Settings ---
INPUT_JSON = '/home/aparcedo/IASEB/clustering/hcstvg1_hcstvg2_vidvrd_vidstg_captions_with_dataset.json'
# SAMPLE_SIZE = 3200
# OUTPUT_CSV = f'hcstvg12_vidvrdstg_gpt4omini_st_class_{SAMPLE_SIZE}sample_v1.csv'
OUTPUT_CSV = f'hcstvg12_vidvrdstg_gpt4omini_st_class_v1.csv'

MODEL_NAME = "gpt-4o-mini"

HIERARCHY = HIERARCHY_V2 = """
    # 1.0 Spatial Relationships (Static): Relationships inferable from a single moment.
        # 1.1 Relative Position (e.g., Near, Beside, InFrontOf, Behind, Above, Below, Between)
        # 1.2 Contact: Interactions involving direct physical touch.
            # 1.2.1 Supportive Contact (e.g., SitsOn, LeansOn, StandsOn, LiesOn, RestsOn)
            # 1.2.2 Manipulative Contact (e.g., Holds, Grabs, Wears, Carries, Uses)
            # 1.2.3 Social/Affectionate Contact (e.g., Hugs, Kisses, HoldsHand, High-fives)
        # 1.3 Perceptual & Indicative Relationships: Non-physical links based on senses or gestures.
            # 1.3.1 Gaze (e.g., Watching, LookingAt, StaresAt, GlancesAt)
            # 1.3.2 Indicative Gesture (e.g., PointsTo, GesturesTowards, NodsAt)
        # 1.4 Communicative Acts (e.g., SpeaksTo, ListensTo, Nods, ShakesHead)

    # 2.0 State Changes & Sequential Actions (Temporal): Relationships defined by change or duration over time, without significant change in relative spatial position.
        # 2.1 Actor State Change (e.g., StandsUp, SitsDown, TurnsAround, BendsOver, UnfastensSeatbelt, Nods, ShakesHead)
        # 2.2 Object State Change (e.g., OpensDoor, ClosesDoor, TurnsOnLight, LightsCigarette)
        # 2.3 Sequential Actions (e.g., Knocks then HandsOver, Bends then CoversMouth, Unfastens then StandsUp)
        2.4 Durational States & Non-Actions (e.g., Waits, Pauses, Hesitates, Sleeps, StandsStill, RemainsSeated, SpeaksTo, ListensTo)

    # 3.0 Spatio-Temporal Interactions (Composite): Combines movement through space with a changing relationship.
        # 3.1 Relative Motion: An entity's movement described in relation to another.
            # 3.1.1 Approach & Depart (e.g., WalksTowards, RunsTo, MovesAwayFrom, BacksAwayFrom)
            # 3.1.2 Passing & Crossing (e.g., WalksPast, CreepsPast, FliesNextTo, JumpsBeneath, DrivesAlongside)
            # 3.1.3 Following & Leading (e.g., Follows, Chases, Leads, Escorts)
        # 3.2 Object Transference (e.g., HandsTo, Gives, PicksUp, PutsDown, TakesOff)
        # 3.3 Instantaneous Motion & Impact (e.g., Hits, Touches, Taps, Kicks, Pushes, Throws, Drops)
        # 3.4 Composite Action Sequences (e.g., WalksTo then SitsOn, ClosesDoor and JumpsIn, TurnsHead and SpeaksTo)
    """


FEW_SHOT_EXAMPLES = FEW_SHOT_EXAMPLES_V2 = """
Here are some examples of correct classifications:

- Caption: "an adult leans on a brown sofa at home."
- Correct Classification: 1.2.1 Supportive Contact

- Caption: "an adult man in black hugs another adult woman in purple."
- Correct Classification: 1.2.3 Social/Affectionate Contact

- Caption: "The man in brown clothes turns his head to the right."
- Correct Classification: 2.1 Actor State Change

- Caption: "The person in the white turban walks to the person in black and stops."
- Correct Classification: 3.1.1 Approach & Depart

- Caption: "The woman hands a phone to the man in black."
- Correct Classification: 3.2 Object Transference

- Caption: "The man in the green clothes closes the door and jumps into the carriage."
- Correct Classification: 3.3 Composite Action Sequences
"""

def classify(caption: str) -> str:
    """
    Uses the OpenAI API with the revised hierarchy and examples.
    """
    system_prompt = f"""
    You are a precise video scene classifier. 
    Your task is to classify a caption into the most specific category from the provided hierarchy. 
    Identity the type of interaction in the scene. 

    Hierarchy:
    {HIERARCHY}

    Use these examples to guide your reasoning:
    {FEW_SHOT_EXAMPLES}

    Respond with the format: "Fine-Grained Category".
    """
    
    user_prompt = f"Caption to classify: \"{caption}\""

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
    df = pd.read_json(INPUT_JSON)

    # if len(df) > SAMPLE_SIZE:
    #     df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    #     # df_sample = df.sample(n=SAMPLE_SIZE).copy()
    # else:
    df_sample = df.copy()

    classifications = []
    for caption in tqdm(df_sample['caption'], desc="Classifying Captions"):
        category_str = classify(caption)
        classifications.append(category_str)
        
    df_sample['category'] = classifications
    
    print(f"\nSaving classified data to '{OUTPUT_CSV}'...")
    df_sample[['caption', 'dataset', 'category']].to_csv(OUTPUT_CSV, index=False)
    

if __name__ == "__main__":
    main()