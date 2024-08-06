import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def load_model_and_tokenizer(checkpoint_path, original_model_path):
    print(f"Loading tokenizer from {original_model_path}")
    print(f"Loading model from {checkpoint_path}")
    
    # Load the tokenizer from the original model
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    
    # Load the model from the checkpoint
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=1000):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

def chat():
    checkpoint_path = "smollm_135m_openhermes/checkpoint-10000"  # Path to your checkpoint
    original_model_path = "HuggingFaceTB/SmolLM-135M"  # Path to the original model
    
    try:
        model, tokenizer = load_model_and_tokenizer(checkpoint_path, original_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Chat with the AI (type 'quit' to exit)")
    chat_history = ""
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        # Construct the prompt
        prompt = f"{chat_history}Instruction: {user_input}\nResponse:"
        
        # Generate and print the response
        response = generate_response(model, tokenizer, prompt)
        print("AI:", response)
        
        # Update chat history
        chat_history += f"Instruction: {user_input}\nResponse: {response}\n"

if __name__ == "__main__":
    chat()