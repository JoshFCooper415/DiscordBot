from torch.utils.data import Dataset
from datasets import load_dataset


class OpenHermesDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=512, val_split=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
       
        full_dataset = load_dataset("teknium/openhermes", split="train")
        full_dataset = full_dataset.filter(lambda x: len(x['instruction']) > 0 and len(x['output']) > 0)
        
        # Split the dataset
        split_dataset = full_dataset.train_test_split(test_size=val_split)
        self.data = split_dataset['train'] if split == 'train' else split_dataset['test']
       
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        instruction = self.data[idx]['instruction']
        output = self.data[idx]['output']
        text = f"Instruction: {instruction}\nResponse: {output}"
        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()
