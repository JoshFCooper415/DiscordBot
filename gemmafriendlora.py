import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset  # Add this import
import os
import csv
import re
from typing import List, Dict
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

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

def verify_gpu():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("CUDA is not available. Training will be on CPU and may be very slow.")

def load_model_and_tokenizer(model_name, auth_token):
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    print("Tokenizer loaded.")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Check for CUDA availability
    device_map = "auto" if torch.cuda.is_available() else None
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        use_auth_token=auth_token,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True
    )
    print("Base model loaded.")
    
    # Prepare the model for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    # Set the pad token to be the same as the EOS token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token:", tokenizer.pad_token)
    
    print("Model and tokenizer loaded with 4-bit quantization and LoRA adapters.")
    
    return model, tokenizer

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

        # Clean the message content
        cleaned_content = clean_message(entry['Content'])
        
        # Skip empty messages
        if not cleaned_content:
            continue

        # Replace username with real name if available, using lowercase for lookup
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
class TrumpTweetsDataset(Dataset):
    def __init__(self, tokenizer, min_favorites=150000, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the dataset
        ds = load_dataset("fschlatt/trump-tweets")
        
        # Filter tweets based on the number of favorites
        self.data = [tweet for tweet in ds['train'] if tweet['favorites'] > min_favorites]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tweet = self.data[idx]
        
        # Format the input text
        input_text = f"Human: What did Trump tweet?\nAssistant: Trump: {tweet['text']}"
        
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
class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, name_mapping, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        chat_log = read_chat_log(file_path)
        self.data = convert_to_conversation_pairs(chat_log, name_mapping)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Manually format the input text
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
def load_auth_token(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()
def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

    verify_gpu()

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model_name = "google/gemma-2b"  # Using the smaller 2B model
    auth_token = load_auth_token("key2.txt")
    
    print("Starting to load model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name, auth_token)
    
    # Print the number of trainable parameters
    model.print_trainable_parameters()
    
    csv_file_path = r"C:\Users\joshf\smolLm\Cerver - D2 and Chill - chamber-of [994776397824409652].csv"
    
    name_mapping_path = "name_mapping.txt"
    try:
        print("Loading name mapping...")
        name_mapping = load_name_mapping(name_mapping_path)
        print("Name mapping loaded successfully.")
    except Exception as e:
        print(f"Error loading name mapping: {e}")
        print("Please check your name_mapping.txt file and try again.")
        return

    print("Creating datasets...")
    chat_dataset = ChatDataset(tokenizer, csv_file_path, name_mapping)
    trump_dataset = TrumpTweetsDataset(tokenizer, min_favorites=5000)  # Adjust min_favorites as needed
    
    # Combine the datasets
    combined_dataset = ConcatDataset([chat_dataset, trump_dataset])
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    print(f"  - Chat dataset size: {len(chat_dataset)}")
    print(f"  - Trump tweets dataset size: {len(trump_dataset)}")

    training_args = TrainingArguments(
        output_dir="./gemma_finetuned",
        num_train_epochs=7,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=300,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
    )

    try:
        print("Starting training...")
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    print("Saving LoRA adapters...")
    model.save_pretrained("./gemma_finetuned_lora2")
    tokenizer.save_pretrained("./gemma_finetuned_lora2")
    print("Training completed and LoRA adapters saved.")

if __name__ == "__main__":
    main()