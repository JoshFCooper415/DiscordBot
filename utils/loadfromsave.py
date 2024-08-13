import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    # Load the pre-trained model and tokenizer
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    # Load the forced save file
    forced_save_path = "./force_save.pt"
    if os.path.exists(forced_save_path):
        forced_save = torch.load(forced_save_path)
        print("Loaded forced save:", forced_save)
    else:
        print(f"Error: {forced_save_path} not found.")
        return

    # Check if the forced save was successful
    if forced_save.get("forced_save", False):
        print("Forced save was successful. Proceeding to save in Hugging Face format.")
        
        # Save the model and tokenizer in Hugging Face format
        output_dir = "./smollm_135m_reddit_hf"
        os.makedirs(output_dir, exist_ok=True)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Model and tokenizer saved in Hugging Face format at {output_dir}")
    else:
        print("Error: Forced save was not successful.")

if __name__ == "__main__":
    main()