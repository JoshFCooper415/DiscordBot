from torch.utils.data import Dataset
from datasets import load_dataset

import csv
import pandas as pd
from typing import List, Dict

from utils import clean_message


class ChatDataset(Dataset):
    def __init__(self, tokenizer, file_path, name_mapping, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        chat_log = self.read_chat_log(file_path)
        self.data = self.convert_chat_log_to_conversation_pairs(chat_log, name_mapping)
    
    def convert_chat_log_to_conversation_pairs(self, chat_log: List[Dict], name_mapping: Dict[str, str]) -> List[Dict]:
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
    
    
    def read_chat_log(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
        
        
        
        
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