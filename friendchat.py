import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import os

def load_name_mapping(file_path: str) -> dict[str, str]:
    name_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    aliases, real_name = line.split(':', 1)
                    real_name = real_name.strip()
                    for alias in aliases.split(','):
                        alias = alias.strip().lower()
                        if alias:
                            name_mapping[alias] = real_name
                except ValueError:
                    print(f"Warning: Invalid line format: {line}")
    return name_mapping

def load_model_and_tokenizer(checkpoint_path: str, original_model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    print(f"Loading tokenizer from {original_model_path}")
    print(f"Loading model from {checkpoint_path}")
   
    try:
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None
   
    specific_checkpoint = os.path.join(checkpoint_path, "")
    print(f"Attempting to load model from: {specific_checkpoint}")
   
    try:
        model = AutoModelForCausalLM.from_pretrained(specific_checkpoint, device_map="auto")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
   
    return model, tokenizer

def generate_response(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, max_length: int = 1000) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
   
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.3,
            do_sample=True
        )
   
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

def chat():
    checkpoint_path = "chat_model_finetuned_final"
    original_model_path = "HuggingFaceTB/SmolLM-135M"
    name_mapping_path = "name_mapping.txt"
   
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, original_model_path)
    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return

    try:
        name_mapping = load_name_mapping(name_mapping_path)
        print("Name mapping loaded successfully")
    except Exception as e:
        print(f"Error loading name mapping: {e}")
        print("Continuing without name mapping")
        name_mapping = {}

    user_name = input("Enter your username: ").strip()
    bot_name = input("Enter the bot's username: ").strip()
    print(f"Chat with {bot_name} (type 'quit' to exit)")
   
    conversation_history = ""
    while True:
        user_message = input(f"{user_name}: ").strip()
        if user_message.lower() == 'quit':
            break
       
        # Map user name if it exists in the mapping
        mapped_user_name = name_mapping.get(user_name.lower(), user_name)
        
        conversation_history += f"{mapped_user_name}: {user_message}\n"
        prompt = f"{conversation_history}{bot_name}:"
       
        # Generate the response
        bot_response = generate_response(model, tokenizer, prompt)
        print(f"{bot_name}: {bot_response}")
       
        conversation_history += f"{bot_name}: {bot_response}\n"

if __name__ == "__main__":
    chat()