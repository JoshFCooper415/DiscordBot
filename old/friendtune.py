import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import os
import csv

from src.utils import clean_message

from typing import List, Dict

from src.utils import load_name_mapping, load_model_and_tokenizer


def read_chat_log(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)


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


class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, name_mapping, max_length=1024):  # Reduced max_length
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

def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU. Training will be slow.")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model_path = "./smollm_135m_openhermes_final"  # Local path to your model
    tokenizer_name = "HuggingFaceTB/SmolLM-135M"  # Tokenizer from HuggingFace
    
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_name)
    model = model.to(device)
    
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
        output_dir="./smollm_friendtuned_30",
        num_train_epochs=50,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./smollm_finetuned_final")
    tokenizer.save_pretrained("./smollm_finetuned_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()