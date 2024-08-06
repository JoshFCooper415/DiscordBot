import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto", torch_dtype=torch.bfloat16)
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_length=100, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def main():
    # Path to your retrained model
    checkpoint_path = "./smollm_135m_retrained_final"
    
    # Load the model and tokenizer
    tokenizer, model = load_model(checkpoint_path)
    
    # Example prompts
    prompts = [
        "Once upon a time in a distant galaxy,",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated_texts = generate_text(tokenizer, model, prompt)
        for i, text in enumerate(generated_texts, 1):
            print(f"Generated text {i}:\n{text}\n")

if __name__ == "__main__":
    main()