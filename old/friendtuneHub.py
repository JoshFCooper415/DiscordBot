import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import os
import csv
import re
from typing import List, Dict

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

def read_chat_log(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

def clean_message(message: str) -> str:
    # Remove attachments
    message = re.sub(r'\[Attachment:.*?\]', '', message)
    
    # Remove links (this regex matches common URL patterns)
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', message)
    
    # Remove extra whitespace
    message = ' '.join(message.split())
    
    return message.strip()

def convert_to_conversation_pairs(chat_log: List[Dict], name_mapping: Dict[str, str]) -> List[Dict]:
    conversation_pairs = []
    current_human = None

    for entry in chat_log:
        if entry['Content'] == 'Joined the server.':
            continue

        cleaned_content = clean_message(entry['Content'])
        
        if not cleaned_content:
            continue

        author = name_mapping.get(entry['Author'].lower(), entry['Author'])
        message = f"{author}: {cleaned_content}"

        if current_human is None:
            current_human = message
        else:
            conversation_pairs.append({
                "human": current_human,
                "assistant": message
            })
            current_human = None

    return conversation_pairs
def load_model_and_tokenizer(model_name):
    print(f"Loading model and tokenizer from Hugging Face: {model_name}")
    
    # For GGUF models, we need to use the original model they're based on
    base_model_name = "microsoft/phi-2"  # Phi-2 is the base model for Cinder-Phi-2
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Set the pad token to be the same as the EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token:", tokenizer.pad_token)
    
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, device_map="auto")
    
    # Resize the token embeddings if necessary
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    print("Model loaded in full precision.")
    
    return model, tokenizer

class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, name_mapping, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        chat_log = read_chat_log(file_path)
        self.data = convert_to_conversation_pairs(chat_log, name_mapping)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input text (human message) and target text (assistant message)
        input_text = f"Human: {item['human']}\nAssistant: {item['assistant']}"
        
        # Tokenize input
        encoding = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels: -100 for padded tokens
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def check_gpu():
    print("Checking GPU availability:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA GPUs available. Please check your PyTorch installation and CUDA setup.")

def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

    check_gpu()

    if not torch.cuda.is_available():
        print("No GPU found. The script will attempt to continue, but training may be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    model_name = "Josephgflowers/Cinder-Phi-2-V1-F16-gguf"
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = model.to(device)  # Explicitly move model to the selected device
    
    csv_file_path = r"C:\Users\joshf\smolLm\Cerver - D2 and Chill - chamber-of [994776397824409652].csv"
    
    name_mapping_path = "name_mapping.txt"
    try:
        name_mapping = load_name_mapping(name_mapping_path)
    except Exception as e:
        print(f"Error loading name mapping: {e}")
        print("Please check your name_mapping.txt file and try again.")
        return

    train_dataset = ChatDataset(tokenizer, csv_file_path, name_mapping)
    
    print(f"Dataset size: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir="./cinder_phi2_finetuned",
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        no_cuda=not torch.cuda.is_available(),  # Disable CUDA if not available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    try:
        if torch.cuda.is_available():
            with autocast():
                trainer.train()
        else:
            trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./cinder_phi2_finetuned_final")
    tokenizer.save_pretrained("./cinder_phi2_finetuned_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()