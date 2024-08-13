import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import os

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

    # Load the fine-tuned model and tokenizer
    checkpoint = "./smollm_135m_reddit_hf"
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} not found. Please ensure the directory exists.")
        return

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
   
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
   
    train_dataset = OpenHermesDataset(tokenizer, split="train")
    val_dataset = OpenHermesDataset(tokenizer, split="test")  # This will use 10% of the data for validation
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir="./smollm_135m_openhermes",
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=2000,
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
        with autocast():
            trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./smollm_135m_openhermes_final")
    tokenizer.save_pretrained("./smollm_135m_openhermes_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()