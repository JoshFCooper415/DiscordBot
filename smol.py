import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.cuda.amp import autocast
from datasets import load_dataset
import os

class StreamingCosmopediaDataset(torch.utils.data.IterableDataset):
    def __init__(self, block_size=128):
        self.block_size = block_size
        self.dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        for item in self.dataset:
            text = item['text']
            tokens = self.tokenizer.encode(text, max_length=self.block_size + 1, truncation=True)
            if len(tokens) < self.block_size + 1:
                continue  # Skip sequences that are too short
            for i in range(0, len(tokens) - self.block_size):
                chunk = tokens[i:i + self.block_size + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {'input_ids': x, 'labels': y}

def get_cosmopedia_loader(batch_size, block_size):
    dataset = StreamingCosmopediaDataset(block_size)
    return DataLoader(dataset, batch_size=batch_size)

class DebugTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            return super().compute_loss(model, inputs, return_outputs)
        except RuntimeError as e:
            print(f"RuntimeError in compute_loss: {e}")
            print(f"Input shapes: {{k: v.shape for k, v in inputs.items()}}")
            raise

def main():
    if not torch.cuda.is_available():
        print("No GPU found. Please use a GPU to train this model.")
        return
    device = torch.device("cuda")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # Enable CUDA error debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Load the pre-trained model and tokenizer
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Create data loaders
    train_loader = get_cosmopedia_loader(batch_size=4, block_size=512)  # Reduced batch size
    val_loader = get_cosmopedia_loader(batch_size=4, block_size=512)  # Reduced batch size

    # Calculate max_steps
    total_train_steps = 10000  # Reduced number of steps

    training_args = TrainingArguments(
        output_dir="./smollm_135m_retrained",
        max_steps=total_train_steps,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,  # Reduced batch size
        gradient_accumulation_steps=4,  # Reduced gradient accumulation steps
        learning_rate=1e-5,  # Reduced learning rate
        warmup_steps=1000,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=False,  # Disabled mixed precision training
        evaluation_strategy="steps",
        eval_steps=500,
        dataloader_num_workers=0,  # Disabled multi-processing
        gradient_checkpointing=True,
    )

    trainer = DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    model.save_pretrained("./smollm_135m_retrained_final")
    tokenizer.save_pretrained("./smollm_135m_retrained_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()