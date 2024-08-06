import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import os
import csv
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

def load_model_and_tokenizer(checkpoint_path, original_model_path):
    print(f"Loading tokenizer from {original_model_path}")
    print(f"Loading model from {checkpoint_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
    
    return model, tokenizer

def read_chat_log(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

# Update this function to use lowercase for author lookup
def convert_to_conversation_pairs(chat_log: List[Dict], name_mapping: Dict[str, str]) -> List[Dict]:
    conversation_pairs = []
    current_human = None

    for entry in chat_log:
        if entry['Content'] == 'Joined the server.':
            continue

        # Replace username with real name if available, using lowercase for lookup
        author = name_mapping.get(entry['Author'].lower(), entry['Author'])
        message = f"{author}: {entry['Content']}"
        
        if entry['Attachments']:
            message += f"\n{author}: [Attachment: {entry['Attachments']}]"

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
    def __init__(self, tokenizer, file_path, name_mapping, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        chat_log = read_chat_log(file_path)
        self.data = convert_to_conversation_pairs(chat_log, name_mapping)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Human: {item['human']}\nAssistant: {item['assistant']}"
        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

def data_collator(features):
    input_ids = torch.stack([f[0] for f in features])
    attention_mask = torch.stack([f[1] for f in features])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone(),
    }

def main():
    if not torch.cuda.is_available():
        print("No GPU found. Please use a GPU to train this model.")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    checkpoint_path = "smollm_135m_openhermes/checkpoint-10000"
    original_model_path = "HuggingFaceTB/SmolLM-135M"
    
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, original_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    csv_file_path = r"C:\Users\joshf\smolLm\Cerver - D2 and Chill - chamber-of [994776397824409652].csv"
    
    # Load the name mapping
    name_mapping_path = "name_mapping.txt"  # You'll need to create this file
    try:
        name_mapping = load_name_mapping(name_mapping_path)
    except Exception as e:
        print(f"Error loading name mapping: {e}")
        print("Please check your name_mapping.txt file and try again.")
        return

    train_dataset = ChatDataset(tokenizer, csv_file_path, name_mapping)
    
    print(f"Dataset size: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir="./chat_model_friendtuned",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    try:
        with autocast():
            trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./chat_model_finetuned_final")
    tokenizer.save_pretrained("./chat_model_finetuned_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()