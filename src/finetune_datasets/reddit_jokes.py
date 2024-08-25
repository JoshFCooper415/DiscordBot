from torch.utils.data import Dataset
from datasets import load_dataset

class RedditJokesDataset(Dataset):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
       
        self.data = load_dataset("SocialGrep/one-million-reddit-jokes", split='train')
        self.data = self.data.filter(lambda x: self.filter_data(x))
       
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        title = self.data[idx]['title'] or ""
        selftext = self.data[idx]['selftext'] or ""
        full_text = f"{title}\n{selftext}".strip()
        encoding = self.tokenizer(full_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

    @staticmethod
    def filter_data(x):
        title = x['title'] or ""
        selftext = x['selftext'] or ""
        return len(title + selftext) > 0