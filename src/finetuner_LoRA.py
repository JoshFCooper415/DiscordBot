import warnings
import torch
from transformers import TrainingArguments, Trainer
from torch.utils.data import ConcatDataset
from datasets import load_dataset  # Add this import
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import json

from utils import show_gpu_specs, load_name_mapping, load_model_and_tokenizer, load_auth_token
from finetune_datasets.trump_tweets import TrumpTweetsDataset
from finetune_datasets.chat_logs import ChatDataset

config = json.load(open("config.json"))

HUGGING_FACE_AUTH_TOKEN_PATH = config["hugging_face_auth_token_file"]
NAME_MAPPING_PATH = config["name_mapping_file"]    
TRAINING_CSV_FILE = "message_histories/Cerver/chamber-of.csv"
    
def main():
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

    show_gpu_specs()

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    model_name = "google/gemma-2-2b-it"  # Using the smaller 2B model
    hugging_face_auth_token = load_auth_token(HUGGING_FACE_AUTH_TOKEN_PATH)
    
    print("Starting to load model and tokenizer...")
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model, tokenizer = load_model_and_tokenizer(model_name, hugging_face_auth_token=hugging_face_auth_token)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    
    # Print the number of trainable parameters
    model.print_trainable_parameters()
    
    csv_file_path = TRAINING_CSV_FILE
    
    try:
        print("Loading name mapping...")
        name_mapping = load_name_mapping(NAME_MAPPING_PATH)
        print("Name mapping loaded successfully.")
    except Exception as e:
        print(f"Error loading name mapping: {e}")
        print("Please check your name_mapping file and try again.")
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
    model.save_pretrained("./gemma_finetuned_lora-2b")
    tokenizer.save_pretrained("./gemma_finetuned_lora-2b")
    print("Training completed and LoRA adapters saved.")

if __name__ == "__main__":
    main()