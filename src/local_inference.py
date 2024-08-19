import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationMixin, PreTrainedTokenizer, QuantoConfig, PreTrainedModel

from typing import List
import asyncio

from utils import load_model_and_tokenizer, async_generate_response, async_redact_text, load_auth_token

os.environ["TOKENIZERS_PARALLELISM"] = "False"


# Constants
# ORIGINAL_MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"
ORIGINAL_MODEL_NAME = "google/gemma-2-2b-it"
# ORIGINAL_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# FINETUNED_MODEL_PATH = "C:/Users/joshf/smolLm/longcustom_finetuned_results/checkpoint-3000"  # Adjust this path as needed
MAX_LENGTH = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
QUANTIZATION_CONFIG = QuantoConfig(weights="int4")


async def main():
    hugging_face_auth_token = load_auth_token("hugging_face_auth_token.txt")
    model, tokenizer = load_model_and_tokenizer(ORIGINAL_MODEL_NAME, QUANTIZATION_CONFIG, hugging_face_auth_token)
    
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    print("Model and tokenizer loaded. Ready for input!")
    
    while True:
        user_input = input("Enter your prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        full_prompt = f"Human: {user_input}\n\nAssistant:"
        response = await async_generate_response(
            model = model, 
            tokenizer = tokenizer, 
            prompt = full_prompt, 
            max_tokens = MAX_LENGTH, 
            temperature = TEMPERATURE, 
            top_p = TOP_P
        )
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())