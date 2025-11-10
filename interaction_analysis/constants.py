# constants.py
import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ---
# --- CENTRAL CONTROL PANEL FOR COLORS, CATEGORIES, & STYLES ---
# ---

# --- Universal Plotting Styles ---
# In constants.py

# --- Universal Plotting Styles ---
UNIVERSAL_FONTSIZE = 18
MIN_FONT_SIZE = 8 
RADAR_REFERENCE_ANGLE = 20 

# --- HSL Color Gradient Generation Parameters ---
# These are used by the create_hsl_colormaps function
HSL_PARAMS = {
    "start_angle_degrees": 30.0,
    "lightness_start": 0.6,
    "lightness_end": 0.85,
    "sat_start": 0.7,
    "sat_end": 0.4,
}

# --- Sunburst Color Sampling ---
INNER_RING_GRADIENT_POS = 0.1
OUTER_RING_GRADIENT_RANGE = [0.7, 1.0]

# ---
# --- CATEGORY DEFINITIONS ---
# ---

DATASET_PATHS = {
    "hcstvg1": {
        "video": "/home/c3-0/datasets/stvg/hcstvg1/v1/video",
        "referral": "/home/c3-0/datasets/stvg/preprocess_dump/hcstvg/hcstvg_pid_tubes_multi_sent_refined_v3/sentences_test.json", 
        "freeform": "/home/c3-0/datasets/stvg/hcstvg1/test_proc.json", 
    }, 
    "hcstvg2": {
        "video": "/home/c3-0/datasets/stvg/hcstvg2/videos",
        "referral": "/home/we337236/stvg/dataset/hcstvg_v2/hcstvgv2_sentences_test_gpt_modified.json", 
        "freeform": "/home/c3-0/datasets/stvg/hcstvg2/annotations/HCVG_val_proc.json", 
    }, 
    "vidstg": {
        "referral": "/share/datasets/stvg/vidstg_annotations/vidstg_referral.json", 
        "freeform": "/home/we337236/stvg/dataset/vidstg/vidstg_pro_test_final_list.json", 
    }, 
    "vidvrd": {
        "referral": "/home/we337236/stvg/dataset/vidvrd/referral_final_output.json", 
        "freeform": "/home/we337236/stvg/dataset/vidvrd/free_form_final_output.json", 
    }, 
    "mevis": {
        "video": "/share/datasets/stvg/MeViS/MeViS/valid_u/JPEGImages/JPEGImages",
        "metadata": "/share/datasets/stvg/mevis_annotations/valid_u/one_object_meta_expressions.json",
        "bbox": "/share/datasets/stvg/mevis_annotations/valid_u/one_obj_bbox_updated_format.json",
        "masks": "/share/datasets/stvg/mevis_annotations/valid_u/mask_dict.json"
    },
    "rvos": {
        "video": "/share/datasets/stvg/rvos_annotations/valid/JPEGImages",
        "masks": "/share/datasets/stvg/rvos_annotations/valid/Annotations",
        "metadata": "/share/datasets/stvg/rvos_annotations/valid/meta_expressions_challenge.json",
        "bbox": "/share/datasets/stvg/rvos_annotations/valid/rvos_bbox_annotations.json",
    }
}


ST_HIERARCHY = {
    1: {'description': 'Spatial Relationships (Static): Relationships inferable from a single moment.',
        'short_name': 'Spatial (Static)',
        'children': {
            1: {'description': 'Relative Position (e.g., Near, Beside, InFrontOf, Behind, Above, Below, Between)', 'short_name': 'Position', 'children': {}},
            2: {'description': 'Contact: Interactions involving direct physical touch.',
                'short_name': 'Contact',
                'children': {
                    1: {'description': 'Supportive Contact (e.g., SitsOn, LeansOn, StandsOn, LiesOn, RestsOn)', 'short_name': 'Supportive', 'children': {}},
                    2: {'description': 'Manipulative Contact (e.g., Holds, Grabs, Wears, Carries, Uses)', 'short_name': 'Manipulative', 'children': {}},
                    3: {'description': 'Social/Affectionate Contact (e.g., Hugs, Kisses, HoldsHand, High-fives)', 'short_name': 'Social', 'children': {}}
                }
            },
            3: {'description': 'Perceptual & Indicative Relationships: Non-physical links based on senses or gestures.',
                'short_name': 'Perception',
                'children': {
                    1: {'description': 'Gaze (e.g., Watching, LookingAt, StaresAt, GlancesAt)', 'short_name': 'Gaze', 'children': {}},
                    2: {'description': 'Indicative Gesture (e.g., PointsTo, GesturesTowards, NodsAt)', 'short_name': 'Gesture', 'children': {}}
                }
            },
            4: {'description': 'Communicative Acts (e.g., SpeaksTo, ListensTo, Nods, ShakesHead)', 'short_name': 'Communication', 'children': {}}
        }
    },
    2: {'description': 'State Changes & Sequential Actions (Temporal): Relationships defined by change or duration over time, without significant change in relative spatial position.',
        'short_name': 'Temporal (State)',
        'children': {
            1: {'description': 'Actor State Change (e.g., StandsUp, SitsDown, TurnsAround, BendsOver, UnfastensSeatbelt, Nods, ShakesHead)', 'short_name': 'Actor State', 'children': {}},
            2: {'description': 'Object State Change (e.g., OpensDoor, ClosesDoor, TurnsOnLight, LightsCigarette)', 'short_name': 'Object State', 'children': {}},
            3: {'description': 'Sequential Actions (e.g., Knocks then HandsOver, Bends then CoversMouth, Unfastens then StandsUp)', 'short_name': 'Sequential', 'children': {}},
            4: {'description': 'Durational States & Non-Actions (e.g., Waits, Pauses, Hesitates, Sleeps, StandsStill, RemainsSeated, SpeaksTo, ListensTo)', 'short_name': 'Durational', 'children': {}}
        }
    },
    3: {'description': 'Spatio-Temporal Interactions (Composite): Combines movement through space with a changing relationship.',
        'short_name': 'Spatio-Temporal',
        'children': {
            1: {'description': "Relative Motion: An entity's movement described in relation to another.",
                'short_name': 'Relative Motion',
                'children': {
                    1: {'description': 'Approach & Depart (e.g., WalksTowards, RunsTo, MovesAwayFrom, BacksAwayFrom)', 'short_name': 'Approach/Depart', 'children': {}},
                    2: {'description': 'Passing & Crossing (e.g., WalksPast, CreepsPast, FliesNextTo, JumpsBeneath, DrivesAlongside)', 'short_name': 'Pass/Cross', 'children': {}},
                    3: {'description': 'Following & Leading (e.g., Follows, Chases, Leads, Escorts)', 'short_name': 'Follow/Lead', 'children': {}}
                }
            },
            2: {'description': 'Object Transference (e.g., HandsTo, Gives, PicksUp, PutsDown, TakesOff)',
                'short_name': 'Transference', 'children': {}},
            3: {'description': 'Instantaneous Motion & Impact (e.g., Hits, Touches, Taps, Kicks, Pushes, Throws, Drops)',
                'short_name': 'Impact',
                'children': {}},
            4: {'description': 'Composite Action Sequences (e.g., WalksTo then SitsOn, ClosesDoor and JumpsIn, TurnsHead and SpeaksTo)',
                'short_name': 'Composite',
                'children': {}}
        }
    }
}

ENTITY_HIERARCHY = {
    1: {'description': 'Human-Human',
        'short_name': 'Human-Human',
        'children': {
            1: {'description': 'Cooperative (e.g., helping, exchanging)', 'short_name': 'Cooperative', 'children': {}},
            2: {'description': 'Competitive (e.g., fighting, sports)', 'short_name': 'Competitive', 'children': {}},
            3: {'description': 'Affective (e.g., hugging, arguing, kissing)', 'short_name': 'Affective', 'children': {}},
            4: {'description': 'Proximity (e.g., person A is behind person B, standing near, away)', 'short_name': 'Proximity', 'children': {}},
            5: {'description': 'Observation (e.g., watching, looking at)', 'short_name': 'Observation', 'children': {}},
            6: {'description': 'Spatial (e.g., human/object A is larger/smaller than human/object B)', 'short_name': 'Spatial', 'children': {}}
        }
    },
    2: {'description': 'Human-Object',
        'short_name': 'Human-Object',
        'children': {
            1: {'description': 'Active Manipulation (e.g., opening, cutting, riding bicycle, holding, carrying)', 'short_name': 'Active Manip.', 'children': {}},
            2: {'description': 'Proximity (e.g., person A is behind object B, standing near, away, pass)', 'short_name': 'Proximity', 'children': {}},
            3: {'description': 'Passive (e.g., sitting, wearing)', 'short_name': 'Passive', 'children': {}},
            4: {'description': 'Spatial (e.g., human/object A is larger/smaller than human/object B)', 'short_name': 'Spatial', 'children': {}}
        }
    },
    3: {'description': 'Human-Animal',
        'short_name': 'Human-Animal',
        'children': {
            1: {'description': 'Direct Interaction (e.g., petting, feeding, touching)', 'short_name': 'Direct Interact.', 'children': {}},
            2: {'description': 'Observation (e.g., watching)', 'short_name': 'Observation', 'children': {}},
            3: {'description': 'Proximity (e.g., standing near)', 'short_name': 'Proximity', 'children': {}},
            4: {'description': 'Spatial (e.g., human/animal A is larger/smaller than human/animal B)', 'short_name': 'Spatial', 'children': {}}
        }
    },
    4: {'description': 'Animal-Animal',
        'short_name': 'Animal-Animal',
        'children': {
            1: {'description': 'Proximity (e.g., playing, standing near, to the right/left of)', 'short_name': 'Proximity', 'children': {}},
            2: {'description': 'Antagonistic (e.g., fighting, hunting)', 'short_name': 'Antagonistic', 'children': {}},
            3: {'description': 'Observation (e.g., watching, looking at)', 'short_name': 'Observation', 'children': {}},
            4: {'description': 'Spatial (e.g., animal A is larger/smaller than animal B)', 'short_name': 'Spatial', 'children': {}}
        }
    },
    5: {'description': 'Animal-Object',
        'short_name': 'Animal-Object',
        'children': {
            1: {'description': 'Interaction (e.g., playing with toy, building nest)', 'short_name': 'Interaction', 'children': {}},
            2: {'description': 'Proximity (e.g., standing near, away)', 'short_name': 'Proximity', 'children': {}},
            3: {'description': 'Spatial (e.g., animal/object A is larger/smaller than animal/object B)', 'short_name': 'Spatial', 'children': {}}
        }
    },
    6: {'description': 'Object-Object',
        'short_name': 'Object-Object',
        'children': {
            1: {'description': 'Spatial/Movement (e.g., car moving near another car)', 'short_name': 'Spatial/Move', 'children': {}},
            2: {'description': 'Proximity (e.g., object A is beneath/above object B, object C is away/close from/to object D)', 'short_name': 'Proximity', 'children': {}},
            3: {'description': 'Spatial (e.g., object A is larger/smaller than object B)', 'short_name': 'Spatial', 'children': {}}
        }
    },
    7: {'description': 'Human-Self Self-interaction (e.g., cover mouth, raise hand)',
        'short_name': 'Human-Self',
        'children': {}},
    8: {'description': 'No Interaction A single agent or object acting in isolation.',
        'short_name': 'No Interaction',
        'children': {}}
}

def create_hsl_colormaps(category_names, 
                         start_angle_degrees=0,
                         lightness_start=0.5,
                         lightness_end=0.5,
                         sat_start=0.2,
                         sat_end=0.6):
    """
    Generates a dict of distinct colormaps based on HSL properties.
    """
    num_categories = len(category_names)
    if num_categories == 0:
        return {}
        
    cmap_dict = {}
    start_hue = start_angle_degrees / 360.0 # Convert degrees (0-360) to HLS scale (0-1)

    for i, category_name in enumerate(category_names):
        hue = (start_hue + (i / float(num_categories))) % 1.0        
        start_rgb = colorsys.hls_to_rgb(hue, lightness_start, sat_start)
        end_rgb = colorsys.hls_to_rgb(hue, lightness_end, sat_end)
        cmap_name = f'hls_grad_{i}'
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, [start_rgb, end_rgb])
        cmap_dict[category_name] = cmap
        
    return cmap_dict

def sample_colormaps(base_cmaps, position):
    """
    Samples each colormap in the dict at a specific position (0-1)
    to create a new dict of single {category_name: (r,g,b,a)} colors.
    """
    return {
        category_name: cmap(position) 
        for category_name, cmap in base_cmaps.items()
    }

# ---
# --- PRE-CALCULATED EXPORTED CONSTANTS ---
# ---

DEFAULT_FALLBACK_CMAP = plt.get_cmap('Greys')

# --- Spatiotemporal (st) Colors ---
st_keys = sorted(ST_HIERARCHY.keys())
ST_CATEGORY_NAMES = [ST_HIERARCHY[key]['short_name'] for key in st_keys]
ST_BASE_COLORMAPS = create_hsl_colormaps(
    ST_CATEGORY_NAMES,
    **HSL_PARAMS
)
# This is the single-color map for sunburst inner ring AND radar outer ring
ST_COARSE_COLOR_MAP = sample_colormaps(
    ST_BASE_COLORMAPS, 
    INNER_RING_GRADIENT_POS
)

# --- Entity Colors ---
entity_keys = sorted(ENTITY_HIERARCHY.keys())
ENTITY_CATEGORY_NAMES = [ENTITY_HIERARCHY[key]['short_name'] for key in entity_keys]
ENTITY_BASE_COLORMAPS = create_hsl_colormaps(
    ENTITY_CATEGORY_NAMES,
    **HSL_PARAMS
)
# This is the single-color map for sunburst inner ring AND radar outer ring
ENTITY_COARSE_COLOR_MAP = sample_colormaps(
    ENTITY_BASE_COLORMAPS, 
    INNER_RING_GRADIENT_POS
)