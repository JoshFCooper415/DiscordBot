import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig, PreTrainedTokenizer, PreTrainedModel

from peft import PeftModel
import traceback
import re

from typing import Dict, Tuple, List


# TODO add options for using bits and bytes quantization
def load_model_and_tokenizer(base_model_path: str, quantization_config: QuantoConfig = None, hugging_face_auth_token: str = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print(f"Loading tokenizer from {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, token = hugging_face_auth_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token: ", tokenizer.pad_token)
    print("Tokenizer loaded successfully")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    
    # Load the base model
    print("Loading base model...")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = base_model_path,
        token = hugging_face_auth_token,
        quantization_config = quantization_config,
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Move model to the appropriate device
    model = model.to(device)
    print(f"Model moved to {device}")
    print("Base model loaded successfully")
    
    return model, tokenizer
    
    
    
def load_lora_adapter_from_base_model(base_model: PreTrainedModel, lora_path: str):
    try:
        # Load the LoRA model
        print("Applying LoRA adapters...")
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        print("LoRA adapters applied successfully")

        return lora_model

    except Exception as e:
        print(f"Error loading lora adapters: {e}")
        traceback.print_exc()
        raise


def load_name_mapping(file_path: str) -> Dict[str, str]:
    name_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            try:
                aliases, real_name = line.split(':', 1)
            except ValueError:
                print(f"Error on line {line_number}: '{line}'. Each line should contain aliases and a real name separated by a colon (:).")
                continue

            real_name = real_name.strip()
            if not real_name:
                print(f"Warning on line {line_number}: No real name provided for aliases '{aliases}'. Skipping this line.")
                continue

            for alias in aliases.split(','):
                alias = alias.strip().lower()
                if alias:
                    name_mapping[alias] = real_name
                else:
                    print(f"Warning on line {line_number}: Empty alias found. Skipping this alias.")

    if not name_mapping:
        raise ValueError("No valid name mappings found in the file. Please check the format of your name_mapping.txt file.")

    return name_mapping



# TODO stop using re.compile and just put the string into the list
def load_filter_words(file_path: str):
    filter_patterns = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()
                if word and not word.startswith('#'):
                    # Convert starred-out words to regex patterns
                    pattern = word.replace('*', r'\w*')
                    # Add word boundaries and make it case-insensitive
                    
                    filter_patterns.append(re.compile(r'\b' + pattern + r'\b', re.IGNORECASE))
                    
        print("Filter patterns loaded successfully")
        return filter_patterns
        
    except Exception as e:
        print(f"Warning: could not load filter patterns: {e}")
        print("Continuing without filter patterns")
        return []

def load_auth_token(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading key: {e}")
        return None



async def async_generate_response(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7, 
    top_p: float = 0.9
    ) -> tuple:
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_token_id = tokenizer.eos_token_id
    newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    
    max_length = input_ids.shape[1] + max_tokens  # Limit total length to input + max_tokens
    
    generated_ids = input_ids.clone()
    
    for _ in range(max_tokens):
        if generated_ids.shape[1] >= max_length:
            break
        
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == eos_token_id or next_token_id.item() == newline_token_id:
                break
    
            
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = response.split(prompt)[-1].strip()  # Remove the prompt from the response
    
    # Remove any content after a line starting with a name followed by a colon
    response_lines = response.split('\n')
    filtered_response_lines = []
    for line in response_lines:
        if re.match(r'^[A-Za-z]+:', line):
            break
        filtered_response_lines.append(line)
    
    response = '\n'.join(filtered_response_lines).strip()

    return response



async def async_clean_message(message: str) -> str:
    # Remove attachments
    message = re.sub(r'\[Attachment:.*?\]', '', message)
    
    # Remove links (this regex matches common URL patterns)
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
    
    # Remove extra whitespace
    message = ' '.join(message.split())
    
    return message.strip()



async def async_redact_text(text: str, filter_patterns: List) -> str:
    for pattern in filter_patterns:
        text = pattern.sub('[REDACTED]', text)
    return text


def show_gpu_specs():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("CUDA is not available. Training will be on CPU and may be very slow.")
