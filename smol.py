import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.cuda.amp import autocast

class PileDataset(Dataset):
    def __init__(self, file_path, tokenizer, split='train', max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
       
        self.data = load_dataset(file_path, "wikitext-103-raw-v1", split=split)
        self.data = self.data.filter(lambda x: len(x['text']) > 0)
        if split == "train":
            self.data = self.data.select(range(0,300000))
       
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        text = self.data[idx]['text']
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

    # Load the pre-trained model and tokenizer
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    train_dataset = PileDataset("wikitext", tokenizer, split="train")
    val_dataset = PileDataset("wikitext", tokenizer, split="validation")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir="./smollm_135m_retrained",
        num_train_epochs=1,
        per_device_train_batch_size=8,
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
        with autocast():
            trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./smollm_135m_retrained_final")
    tokenizer.save_pretrained("./smollm_135m_retrained_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()