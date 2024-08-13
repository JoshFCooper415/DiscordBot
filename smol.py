import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset, random_split
from torch.amp import autocast
import os

class CustomDataset(TorchDataset):
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the sentence-transformers/reddit dataset
        try:
            self.data = load_dataset("sentence-transformers/reddit", split="train")
            
            print(f"Dataset info: {self.data}")
            print(f"Number of samples: {len(self.data)}")
            print(f"First sample: {self.data[0] if len(self.data) > 0 else 'No samples'}")
            
            if len(self.data) == 0:
                raise ValueError("The dataset is empty.")
            
            if 'title' not in self.data.column_names or 'body' not in self.data.column_names:
                raise KeyError("The dataset does not contain 'title' and 'body' columns.")
        
        except Exception as e:
            print(f"Error loading the dataset: {e}")
            raise
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        title = self.data[idx]['title']
        body = self.data[idx]['body']
        full_text = f"Title: {title}\n\nBody: {body}"
        encoding = self.tokenizer(full_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

def data_collator(features):
    input_ids = torch.stack([f[0] for f in features])
    attention_mask = torch.stack([f[1] for f in features])
    
    # Shift the input_ids to create labels for next token prediction
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Ignore the last token as there's no next token to predict
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

def main():
    if not torch.cuda.is_available():
        print("No GPU found. Please use a GPU to train this model.")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Load the pre-trained model and tokenizer
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # Create the dataset
    try:
        full_dataset = CustomDataset(tokenizer)
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    # Split the dataset
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir="./smollm_135m_reddit",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=5000,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=5000,
        save_total_limit=2,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    try:
        with autocast(device_type='cuda'):
            trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./smollm_135m_reddit")
    tokenizer.save_pretrained("./smollm_135m_reddit")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()