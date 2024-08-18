import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationMixin, PreTrainedTokenizer, QuantoConfig, PreTrainedModel

from typing import List
import asyncio

from utils import load_model_and_tokenizer, generate_response, redact_text, load_auth_token

os.environ["TOKENIZERS_PARALLELISM"] = "False"


# Constants
ORIGINAL_MODEL_NAME = "google/gemma-2-2b-it"
# ORIGINAL_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# FINETUNED_MODEL_PATH = "C:/Users/joshf/smolLm/longcustom_finetuned_results/checkpoint-3000"  # Adjust this path as needed
MAX_LENGTH = 20
TEMPERATURE = 0.7
TOP_P = 0.9


# def generate_response(model: GenerationMixin, tokenizer: PreTrainedTokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float):
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
#     if torch.cuda.is_available():
#         inputs = inputs.to("cuda")
    
#     # de
#     # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
    
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.split("Assistant:", 1)[-1].strip()




async def main():
    hugging_face_auth_token = load_auth_token("hugging_face_auth_token.txt")
    model, tokenizer = load_model_and_tokenizer(ORIGINAL_MODEL_NAME, QuantoConfig(weights="int4"), hugging_face_auth_token)
    
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    print("Model and tokenizer loaded. Ready for input!")
    
    while True:
        user_input = input("Enter your prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        full_prompt = f"Human: {user_input}\n\nAssistant:"
        response = await generate_response(model, tokenizer, full_prompt, MAX_LENGTH, TEMPERATURE, TOP_P)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())