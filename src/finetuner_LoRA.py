import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset  # Add this import
import csv
import re
from typing import List, Dict
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from utils import verify_gpu, load_name_mapping, clean_message
from finetune_datasets.trump_tweets import TrumpTweetsDataset
from finetune_datasets.chat_logs import ChatDataset


def load_model_and_tokenizer(model_name, auth_token):
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token)
    print("Tokenizer loaded.")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Check for CUDA availability
    device_map = "auto" if torch.cuda.is_available() else None
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        use_auth_token=auth_token,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True
    )
    print("Base model loaded.")
    
    # Prepare the model for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    # Set the pad token to be the same as the EOS token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token:", tokenizer.pad_token)
    
    print("Model and tokenizer loaded with 4-bit quantization and LoRA adapters.")
    
    return model, tokenizer





def load_auth_token(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()
    
def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

    verify_gpu()

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model_name = "google/gemma-2-9b-it"  # Using the smaller 2B model
    auth_token = load_auth_token("key2.txt")
    
    print("Starting to load model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name, auth_token)
    
    # Print the number of trainable parameters
    model.print_trainable_parameters()
    
    csv_file_path = r"C:\Users\joshf\smolLm\Cerver - D2 and Chill - chamber-of [994776397824409652].csv"
    
    name_mapping_path = "name_mapping.txt"
    try:
        print("Loading name mapping...")
        name_mapping = load_name_mapping(name_mapping_path)
        print("Name mapping loaded successfully.")
    except Exception as e:
        print(f"Error loading name mapping: {e}")
        print("Please check your name_mapping.txt file and try again.")
        return

    print("Creating datasets...")
    chat_dataset = ChatDataset(tokenizer, csv_file_path, name_mapping)
    trump_dataset = TrumpTweetsDataset(tokenizer, min_favorites=5000)  # Adjust min_favorites as needed
    
    # Combine the datasets
    combined_dataset = ConcatDataset([chat_dataset, trump_dataset])
    
    print(f"Combined dataset size: {len(combined_dataset)}")
    print(f"  - Chat dataset size: {len(chat_dataset)}")
    print(f"  - Trump tweets dataset size: {len(trump_dataset)}")

    training_args = TrainingArguments(
        output_dir="./gemma_finetuned-9b",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=300,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
    )

    try:
        print("Starting training...")
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    print("Saving LoRA adapters...")
    model.save_pretrained("./gemma_finetuned_lora-9b")
    tokenizer.save_pretrained("./gemma_finetuned_lora-9b")
    print("Training completed and LoRA adapters saved.")

if __name__ == "__main__":
    main()