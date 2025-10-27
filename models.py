import re
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, CLIPImageProcessor

# Shikra imports
from mmengine import Config
from transformers import BitsAndBytesConfig
from shikra.mllm.dataset.process_function import PlainBoxFormatter
from shikra.mllm.dataset.builder import prepare_interactive
from shikra.mllm.models.builder.build_shikra import load_pretrained_shikra
from shikra.mllm.dataset.utils.transform import expand2square

# Ferret imports
from ml_ferret.ferret.model import FERRETLlamaForCausalLM
from ml_ferret.ferret.conversation import conv_templates
from ml_ferret.ferret.mm_utils import process_images, tokenizer_image_token
from ml_ferret.ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# --------------------------
# CONFIGURATION
# --------------------------
COGVLM_MODEL_PATH = "zai-org/cogvlm-grounding-generalist-hf"
COGVLM_TOKENIZER_PATH = "lmsys/vicuna-7b-v1.5"
SHIKRA_MODEL_PATH = "/home/aparcedo/shikra/shikras/shikra-7b"
FERRET_MODEL_PATH = "/home/aparcedo/IASEB/ml_ferret/ferret-7b-v1-3"
DEVICE = "cuda"

class FerretSingleSample:
    """
    A class to handle loading the Ferret model and running single-sample inference.
    """
    def __init__(self):
        """
        Initializes the Ferret model, tokenizer, and image processor.
        """
        print(f"Loading Ferret model from {FERRET_MODEL_PATH}...")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(FERRET_MODEL_PATH, use_fast=False)

        # Load the main model
        self.model = FERRETLlamaForCausalLM.from_pretrained(
            FERRET_MODEL_PATH,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        # Load image processor
        print("Loading image processor...")
        vision_tower_name = self.model.config.mm_vision_tower
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name, torch_dtype=torch.float16)

        # Explicitly load vision tower weights and move to the correct device
        print("Explicitly loading vision tower weights...")
        self.model.get_vision_tower().load_model()
        self.model.get_vision_tower().to(device=DEVICE, dtype=torch.float16)
        
        self.model.eval()
        print("Ferret model loaded successfully!")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question.

        Returns:
            tuple: (text_output, boxes, query, response) to match the evaluation script's interface.
        """
        # 1. Prepare the conversation prompt
        conv = conv_templates["ferret_v1"].copy()
        text_prompt = "What is the location of " + question + "?" + f'\n{DEFAULT_IMAGE_TOKEN}'
        conv.append_message(conv.roles[0], text_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 2. Process image and tokenize prompt
        image_tensor = process_images([image.convert('RGB')], self.image_processor, self.model.config).to(DEVICE, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(DEVICE)

        # 3. Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 4. Decode the output
        # import code; code.interact(local=locals())
        response = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f'Prompt: {prompt}')
        print(f"\nFerret Response: {response}")

        # 5. Parse bounding boxes from the response
        text_output = response
        boxes = []
        # Ferret outputs boxes like '... a bird at [260, 335, 405, 513].'
        match = re.search(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', response)
        if match:
            try:
                coords_str = match.group(1).split(',')
                # Coords are already in the 0-1000 range, as expected by the eval script
                coords = [int(c.strip()) for c in coords_str]
                if len(coords) == 4:
                    box_tensor = torch.tensor(coords, device=DEVICE).unsqueeze(0)
                    boxes.append(box_tensor)
            except (ValueError, IndexError):
                print("Could not parse coordinates from Ferret response.")

        return text_output, boxes, text_prompt, response

class ShikraSingleSample:
    def __init__(self, model_path=SHIKRA_MODEL_PATH, load_in_8bit=False):
        """
        Initializes the Shikra model, tokenizer, and preprocessor.
        """
        # Define model configuration from the inference script
        model_args = Config(dict(
            type='shikra',
            version='v1',
            cache_dir=None,
            model_name_or_path=model_path,
            vision_tower=r'openai/clip-vit-large-patch14',
            pretrain_mm_mlp_adapter=None,
            mm_vision_select_layer=-2,
            model_max_length=512,
            freeze_backbone=False,
            tune_mm_mlp_adapter=False,
            freeze_mm_mlp_adapter=False,
            is_multimodal=True,
            sep_image_conv_front=False,
            image_token_len=256,
            mm_use_im_start_end=True,
            target_processor=dict(boxes=dict(type='PlainBoxFormatter')),
            process_func_args=dict(
                conv=dict(type='ShikraConvProcess'),
                target=dict(type='BoxFormatProcess'),
                text=dict(type='ShikraTextProcess'),
                image=dict(type='ShikraImageProcessor'),
            ),
            conv_args=dict(
                conv_template='vicuna_v1.1',
                transforms=dict(type='Expand2square'),
                tokenize_kwargs=dict(truncation_size=None),
            ),
        ))
        
        training_args = Config(dict(bf16=False, fp16=True, device=DEVICE, fsdp=None))
        
        quantization_kwargs = {}
        if load_in_8bit:
            quantization_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

        print("Loading Shikra model...")
        self.model, self.preprocessor = load_pretrained_shikra(model_args, training_args, **quantization_kwargs)
        
        # Move model components to the correct device and dtype
        if not getattr(self.model, 'is_quantized', False):
            self.model.to(dtype=torch.float16, device=torch.device(DEVICE))
        if hasattr(self.model.model, 'vision_tower') and not getattr(self.model.model.vision_tower[0], 'is_quantized', False):
            self.model.model.vision_tower[0].to(dtype=torch.float16, device=torch.device(DEVICE))
        
        self.preprocessor['target'] = {'boxes': PlainBoxFormatter()}
        self.tokenizer = self.preprocessor['text']
        self.model_args = model_args
        print("Shikra model loaded successfully.")

    def run_inference(self, image: Image.Image, question: str):
        """
        Runs inference on a single image and question, returning the parsed bounding box.
        """
        prompt = f'Would you kindly provide the coordinates of {question} located in the picture?'
        processed_image = expand2square(image.convert("RGB"))

        # Prepare conversational input
        ds = prepare_interactive(self.model_args, self.preprocessor)
        ds.set_image(processed_image)
        ds.append_message(role=ds.roles[0], message=prompt)
        
        model_inputs = ds.to_model_input()
        for k, v in model_inputs.items():
            if torch.is_tensor(v):
                model_inputs[k] = v.to(DEVICE)
        
        if 'images' in model_inputs and model_inputs['images'] is not None:
            model_inputs['images'] = model_inputs['images'].to(dtype=torch.float16)

        # Define generation arguments
        gen_kwargs = dict(
            use_cache=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=64,
        )

        # Run model generation
        with torch.inference_mode():
            with torch.autocast(dtype=torch.float16, device_type='cuda'):
                output_ids = self.model.generate(**model_inputs, **gen_kwargs)

        # Decode and parse the response
        input_token_len = model_inputs['input_ids'].shape[-1]
        response_ids = output_ids[:, input_token_len:]
        response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
        print(f"\nShikra Response: {response}")

        text_output = response
        boxes = []
        
        # Parse the bounding box from the response string, e.g., "Here it is [100, 200, 300, 400]"
        match = re.search(r'\[(\d+\.?\d*,\s*\d+\.?\d*,\s*\d+\.?\d*,\s*\d+\.?\d*)\]', response)
        if match:
            try:
                coords_str = match.group(1).split(',')

                # Multiply by 1000 because Shikra output coords in 0..1 range.
                # Rest of the script uses 0..1000 range
                coords = [(float(c.strip()) * 1000) for c in coords_str]
                print(f'coords: {coords}')
                if len(coords) == 4:
                    box_tensor = torch.tensor(coords, device=DEVICE).unsqueeze(0)
                    boxes.append(box_tensor)
            except (ValueError, IndexError):
                print("Could not parse coordinates from Shikra response.")

        return text_output, boxes, prompt, response

class CogVLMSingleSample:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            COGVLM_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(DEVICE).eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(COGVLM_TOKENIZER_PATH)

    def run_inference(self, image, question: str):
        question = question.strip().replace('\\', '')
        query = f'Detect and provide coordinates for "{question}"'

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=[],
            images=[image]
        )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch.bfloat16)]],
        }

        gen_kwargs = {"max_new_tokens": 1024, "do_sample": False}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nCogVLM Response (x0, y0, x1, y1):", response) 

        text_output = response
        boxes = []
        
        match = re.search(r'\[\[(\d+\.?\d*,\d+\.?\d*,\d+\.?\d*,\d+\.?\d*)\]\]', response)
        if match:
            try:
                # Extract the numbers, split by comma, and convert to floats
                coords = [int(c) for c in match.group(1).split(',')]
                if len(coords) == 4:
                    box_tensor = torch.tensor(coords, device=DEVICE).unsqueeze(0)
                    boxes.append(box_tensor)
            except (ValueError, IndexError):
                print("Could not parse coordinates from model response.")
                pass # Failed to parse, boxes remains empty

        return text_output, boxes, query, response