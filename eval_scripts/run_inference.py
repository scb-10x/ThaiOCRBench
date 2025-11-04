import argparse
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from google.genai import types
from google import genai
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["qwen7b", "easy_ocr", "pytesseract", "qwen3b", "gpt4o","claude","gemini", "gemma4b", "gemma12b", "gemma1b" ,"gemma27b","llama11b"], required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def run_qwen_vllm(model_path, dataset, output_path, max_tokens):
    print(f"Loading processor and model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=[]
    )

    predictions = []

    
    for data in tqdm(dataset):
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": data["question"]}
                ],
            }
        ]

        prompt = processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(image_messages)
        mm_data = {"image": image_inputs} if image_inputs else {}

        llm_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        prediction = outputs[0].outputs[0].text

        predictions.append({
            "type": data["Task"],
            "id": data["Id"],
            "question": data["question"],
            "answers": [data["answer"]],
            "predict": prediction
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


def run_easy_ocr(model_path, dataset, output_path, max_tokens):
    print(f"Loading processor and model from {model_path}")
    import easyocr
    reader = easyocr.Reader(['en','th'])
    
    list_ocr = ['Text recognition', 'Full-page OCR', 'Handwritten content extraction']
    
    predictions = []
    for data in tqdm(dataset):
        if data['Task'] not in list_ocr:
            continue
        
        result = reader.readtext(data["image"])
        text_all = ""
        for (bbox, text, prob) in result:
            text_all = text_all + " " + text

        prediction = text_all.strip()

        predictions.append({
            "type": data["Task"],
            "id": data["Id"],
            "question": data["question"],
            "answers": [data["answer"]],
            "predict": prediction
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        

def run_pytesseract(model_path, dataset, output_path, max_tokens):
    print(f"Loading processor and model from {model_path}")
    import pytesseract
    from PIL import Image
    
    list_ocr = ['Text recognition', 'Full-page OCR', 'Handwritten content extraction']
    
    predictions = []
    for data in tqdm(dataset):
        if data['Task'] not in list_ocr:
            continue
        
         
        img = data["image"]
        custom_config = '-c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(img, lang='eng+tha', config=custom_config)

        prediction = text.strip()

        predictions.append({
            "type": data["Task"],
            "id": data["Id"],
            "question": data["question"],
            "answers": [data["answer"]],
            "predict": prediction
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


def run_gemma3_vllm(model_path, dataset, output_path, max_tokens):
    print(f"Loading processor and model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.5,
        max_tokens=max_tokens,
        stop_token_ids=[],
    )

    import base64
    from io import BytesIO
    from PIL import Image



    predictions = []
    all_prompt = []
    data_ref = []
    for data in tqdm(dataset):
        image_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Your task is to answer the question based on provided image."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": data["question"]}
                ],
            }
        ]

        prompt = processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
        
        prompt_in ={
                    "prompt": prompt,
                    "multi_modal_data": {"image": data["image"]},
                }
        
        all_prompt.append(prompt_in)
        data_ref.append({"Task": data["Task"],"Id": data["Id"],"question": data["question"],"answer": data["answer"]})
        
    outputs = llm.generate(all_prompt, sampling_params=sampling_params)
    for resp, data1 in tqdm(zip(outputs, data_ref), total=len(outputs)):
        prediction = resp.outputs[0].text

        predictions.append({
            "type": data1["Task"],
            "id": data1["Id"],
            "question": data1["question"],
            "answers": [data1["answer"]],
            "predict": prediction
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


def run_llama3_vllm(model_path, dataset, output_path, max_tokens):
    print(f"Loading processor and model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=2,
        gpu_memory_utilization=0.6,
        dtype="bfloat16"
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=[]
    )

    predictions = []
    all_prompt = []
    data_ref = []
    for data in tqdm(dataset):
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": data["question"]}
                ],
            }
        ]

        prompt = processor.apply_chat_template(image_messages, tokenize=False, add_generation_prompt=True)
        all_prompt.append(prompt)
        data_ref.append({"Task": data["Task"],"Id": data["Id"],"question": data["question"],"answer": data["answer"]})
        
    outputs = llm.generate(all_prompt, sampling_params=sampling_params)
    for resp, data1 in tqdm(zip(outputs, data_ref), total=len(outputs)):
        prediction = resp.outputs[0].text

        predictions.append({
            "type": data1["Task"],
            "id": data1["Id"],
            "question": data1["question"],
            "answers": [data1["answer"]],
            "predict": prediction
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


import os
import json
import time
import base64
from io import BytesIO
from tqdm import tqdm
from collections import Counter
import anthropic
from PIL import Image

def encode_image_to_base64(image_path, max_size_mb=5):
    """Convert image file to base64 string with size optimization"""
    with open(image_path, "rb") as image_file:
        # Check file size first
        file_size = os.path.getsize(image_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size <= max_size_bytes:
            # File is already small enough
            return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # File is too large, need to process it
            pil_image = Image.open(image_path)
            return encode_pil_image_to_base64(pil_image, max_size_mb)

def encode_pil_image_to_base64(pil_image, max_size_mb=4.8):
    """Convert PIL Image to base64 string with smart aspect ratio-based resizing"""
    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    max_size_bytes = int(max_size_mb * 1024 * 1024)  # Convert MB to bytes with buffer
    
    # Get original dimensions
    original_width, original_height = pil_image.size
    aspect_ratio = original_width / original_height
    total_pixels = original_width * original_height
    
    print(f"Original image: {original_width}x{original_height} (aspect ratio: {aspect_ratio:.2f})")
    
    # Calculate smart resize targets based on aspect ratio and total pixels
    # Target different pixel counts while maintaining aspect ratio
    target_pixel_counts = [
        2048 * 2048,  # ~4MP
        1536 * 1536,  # ~2.3MP  
        1024 * 1024,  # ~1MP
        800 * 800,    # ~640K
        600 * 600,    # ~360K
        400 * 400,    # ~160K
    ]
    
    def calculate_dimensions(target_pixels, aspect_ratio):
        """Calculate width and height for target pixel count maintaining aspect ratio"""
        if aspect_ratio >= 1:  # Landscape or square
            height = int((target_pixels / aspect_ratio) ** 0.5)
            width = int(height * aspect_ratio)
        else:  # Portrait
            width = int((target_pixels * aspect_ratio) ** 0.5)
            height = int(width / aspect_ratio)
        return width, height
    
    # Try original size first with different qualities
    current_image = pil_image.copy()
    
    for quality in [85, 75, 65, 55, 45, 35, 25]:
        buffer = BytesIO()
        current_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        
        buffer_size = len(buffer.getvalue())
        print(f"Original size, Quality {quality}: {buffer_size} bytes ({buffer_size/1024/1024:.2f} MB)")
        
        if buffer_size <= max_size_bytes:
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # If original is too large, try different pixel counts
    for target_pixels in target_pixel_counts:
        if target_pixels >= total_pixels:
            continue  # Skip if target is larger than original
        
        new_width, new_height = calculate_dimensions(target_pixels, aspect_ratio)
        
        # Skip if dimensions are too similar to what we already tried
        if abs(new_width - original_width) < 50 and abs(new_height - original_height) < 50:
            continue
        
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Trying smart resize: {new_width}x{new_height} (~{target_pixels/1000000:.1f}MP)")
        
        for quality in [85, 75, 65, 55, 45, 35, 25]:
            buffer = BytesIO()
            resized_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            
            buffer_size = len(buffer.getvalue())
            print(f"  Quality {quality}: {buffer_size} bytes ({buffer_size/1024/1024:.2f} MB)")
            
            if buffer_size <= max_size_bytes:
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Last resort - very conservative resize
    final_width, final_height = calculate_dimensions(200 * 200, aspect_ratio)
    final_image = pil_image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    final_image.save(buffer, format='JPEG', quality=20, optimize=True)
    buffer.seek(0)
    final_size = len(buffer.getvalue())
    print(f"Final fallback: {final_width}x{final_height}, quality 20: {final_size} bytes ({final_size/1024/1024:.2f} MB)")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_image_media_type(image_path):
    """Get the media type based on file extension"""
    ext = image_path.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg']:
        return 'image/jpeg'
    elif ext == 'png':
        return 'image/png'
    elif ext == 'gif':
        return 'image/gif'
    elif ext == 'webp':
        return 'image/webp'
    else:
        return 'image/jpeg'  # default

def run_claude(dataset, output_path):
    # Initialize Claude client
    client = anthropic.Anthropic(
        api_key=""
    )
    
    # Load existing predictions if file exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        predictions = []

    # Create a set of already processed IDs
    existing_ids = {pred["id"] for pred in predictions}
    
    missing_ids = [id_ for id_ in dataset["Id"] if id_ not in existing_ids]
    
    id_counts = Counter(dataset["Id"])
    # Get only duplicated Ids (count > 1)
    duplicate_ids = [id_ for id_, count in id_counts.items() if count > 1]
    
    for data in tqdm(dataset):
        try:
            if data["Id"] in existing_ids:
                continue  # Skip if already processed
            
            # Prepare the message content
            message_content = []
            # Add text content
            message_content.append({
                "type": "text",
                "text": data['question']
            })
            
            # Add image content if present
            if 'image' in data and data['image']:
                # Check if image is a file path, PIL Image, or already base64 encoded
                if isinstance(data['image'], str):
                    if os.path.isfile(data['image']):
                        # It's a file path, encode it
                        image_base64 = encode_image_to_base64(data['image'])
                        media_type = get_image_media_type(data['image'])
                    else:
                        # Assume it's already base64 encoded
                        image_base64 = data['image']
                        media_type = 'image/jpeg'  # default
                elif hasattr(data['image'], 'mode') and hasattr(data['image'], 'size'):
                    # It's a PIL Image object
                    print(f"Processing PIL Image for ID {data['Id']}, size: {data['image'].size}")
                    image_base64 = encode_pil_image_to_base64(data['image'])
                    media_type = 'image/jpeg'  # We convert PIL images to JPEG
                else:
                    # Handle other image formats
                    print(f"Warning: Unexpected image format for ID {data['Id']}: {type(data['image'])}")
                    break
                
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64
                    }
                })
            
            # Make API call to Claude
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20000,  # Adjust as needed
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ]
            )
            
            # Extract response text
            response_text = response.content[0].text
            
            predictions.append({
                "type": data["Task"],
                "id": data["Id"],
                "question": data["question"],
                "answers": [data["answer"]],
                "predict": response_text
            })
            
            # Save predictions after each successful inference
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            
            # Rate limiting - Claude has different rate limits than Gemini
            time.sleep(1)  # Adjust based on your rate limits
            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print(f"Error processing ID: {data['Id']}")
            # Continue processing other items instead of breaking
            break

def run_gemini(dataset, output_path):
    key_id = ["key1", "key2"] #if you have multiple keys, you can add here
    
    for key_ea in key_id:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", key_ea))

        # Load existing predictions if file exists
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        else:
            predictions = []
        existing_ids = {pred["id"] for pred in predictions}

        missing_ids = [id_ for id_ in dataset["Id"] if id_ not in existing_ids]

        from collections import Counter
        id_counts = Counter(dataset["Id"])

        # Get only duplicated Ids (count > 1)
        duplicate_ids = [id_ for id_, count in id_counts.items() if count > 1]
        
        for data in tqdm(dataset):
            try:
                if data["Id"] in existing_ids:
                    #print("Skip ",data["Id"])
                    continue  # Skip if already processed
                
                config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=40000)
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[data['question'], data['image']],
                    config=config
                )
                
                predictions.append({
                    "type": data["Task"],
                    "id": data["Id"],
                    "question": data["question"],
                    "answers": [data["answer"]],
                    "predict": response.text
                })

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                time.sleep(2)  # Respect Gemini API rate limit
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                break
        #break

import os
import time
import json
import base64
from io import BytesIO
from tqdm import tqdm
from collections import Counter
from openai import OpenAI  # Requires openai>=1.3.5

def pil_image_to_base64(pil_img):
    """Convert a PIL image to base64 JPEG string"""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def run_gpt4o(dataset, output_path):
    api_key1 = ""
    
    client = OpenAI(api_key=api_key1)

    # Load previous predictions
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        predictions = []

    existing_ids = {pred["id"] for pred in predictions}

    id_counts = Counter(dataset["Id"])
    duplicate_ids = [id_ for id_, count in id_counts.items() if count > 1]

    for data in tqdm(dataset):
        try:
            if data["Id"] in existing_ids:
                continue

            # Build user content block with text and optional image
            user_content = [{"type": "text", "text": data["question"]}]

            if 'image' in data and data['image']:
                if hasattr(data['image'], 'mode') and hasattr(data['image'], 'size'):
                    b64_image = pil_image_to_base64(data['image'])
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": b64_image}
                    })
                else:
                    print(f"Warning: unexpected image type for ID {data['Id']}")
                    continue

            # GPT-4o API call using preferred message format
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": user_content}
                ],
                max_tokens=16384,
                temperature=0.0
            )

            response_text = completion.choices[0].message.content

            predictions.append({
                "type": data["Task"],
                "id": data["Id"],
                "question": data["question"],
                "answers": [data["answer"]],
                "predict": response_text
            })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

            time.sleep(1)

        except Exception as err:
            print(f"Unexpected error: {err} ({type(err)})")
            print(f"Error processing ID: {data['Id']}")
            break


if __name__ == "__main__":
    args = get_args()

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    # Set env vars
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['HF_TOKEN'] = args.hf_token
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'  # Use FlashInfer if available

    print("Loading dataset...")
    dataset = load_dataset("scb10x/ThaiOCRBench", token=args.hf_token)['test']
    
    dataset = dataset.map(lambda x: {
            "question": "Instruction: " + x["question"].strip() + "\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม" if isinstance(x["question"], str) else x["question"],
            "answer": x["answer"].strip() if isinstance(x["answer"], str) else x["answer"]
        })

    if args.max_samples:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    
    if args.model_name == "gemini":
        run_gemini(dataset, args.output_path)
    elif args.model_name == "claude":
        run_claude(dataset, args.output_path)
    elif args.model_name == "gpt4o":
        run_gpt4o(dataset, args.output_path)
    elif args.model_name == "qwen7b":
        run_qwen_vllm("Qwen/Qwen2.5-VL-7B-Instruct", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "qwen3b":
        run_qwen_vllm("Qwen/Qwen2.5-VL-3B-Instruct", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "gemma4b":
        run_gemma3_vllm("google/gemma-3-4b-it", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "gemma27b":
        run_gemma3_vllm("google/gemma-3-27b-it", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "gemma12b":
        run_gemma3_vllm("google/gemma-3-12b-it", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "llama11b":
        run_llama3_vllm("meta-llama/Llama-3.2-11B-Vision-Instruct", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "easy_ocr":
        run_easy_ocr("", dataset, args.output_path, max_tokens=11000)
    elif args.model_name == "pytesseract":
        run_pytesseract("", dataset, args.output_path, max_tokens=11000)

