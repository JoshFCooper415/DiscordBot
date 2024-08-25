from torch.utils.data import Dataset
from datasets import load_dataset

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
        