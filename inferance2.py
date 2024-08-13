import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
ORIGINAL_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
FINETUNED_MODEL_PATH = "C:/Users/joshf/smolLm/longcustom_finetuned_results/checkpoint-3000"  # Adjust this path as needed
MAX_LENGTH = 2048
TEMPERATURE = 0.7
TOP_P = 0.9

def load_model_and_tokenizer():
    print("Loading the tokenizer and model...")
    try:
        # Load the tokenizer from the original model
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
        
        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, torch_dtype=torch.bfloat16)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:", 1)[-1].strip()

def main():
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    print("Model and tokenizer loaded. Ready for input!")
    
    while True:
        user_input = input("Enter your prompt (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        full_prompt = f"Human: {user_input}\n\nAssistant:"
        response = generate_response(model, tokenizer, full_prompt)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()