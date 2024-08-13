import discord
from discord.ext import commands
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
import asyncio
import re
import datetime

class InferenceBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(
            command_prefix='!', 
            intents=intents,
            permissions=discord.Permissions(3072)
        )
        self.model = None
        self.tokenizer = None
        self.name_mapping = {}
        self.filter_patterns = []

    async def load_model_and_tokenizer(self, base_model_path: str, lora_path: str):
        print(f"Loading tokenizer from {base_model_path}")
        print(f"Loading LoRA adapters from {lora_path}")
       
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            print("Tokenizer loaded successfully")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                print("CUDA is available. Using GPU.")
                device = torch.device("cuda")
            else:
                print("CUDA is not available. Using CPU.")
                device = torch.device("cpu")
            
            # Load the base model
            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("Base model loaded successfully")
            
            # Load the LoRA configuration
            print("Loading LoRA configuration...")
            peft_config = PeftConfig.from_pretrained(lora_path)
            print("LoRA configuration loaded successfully")
            
            # Load the LoRA model
            print("Applying LoRA adapters...")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA adapters applied successfully")
            
            # Move model to the appropriate device
            self.model = self.model.to(device)
            print(f"Model moved to {device}")
            
            print("Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            import traceback
            traceback.print_exc()
            raise
    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith('!'):
            await self.handle_command(message)
        
        await self.process_commands(message)

    async def handle_command(self, message):
        parts = message.content[1:].split(maxsplit=1)
        if len(parts) == 2:
            target_name, content = parts
            target_name = target_name.lower()
            
            if target_name in self.name_mapping:
                real_target_name = self.name_mapping[target_name]
            else:
                real_target_name = target_name.capitalize()
            
            user_name = self.name_mapping.get(message.author.name.lower(), message.author.name)
            
            prompt = f"{user_name}: {content}\n{real_target_name}:"
            
            try:
                async with message.channel.typing():
                    unredacted_response, redacted_response = await self.generate_response(prompt)
                
                # Send the message
                await message.channel.send(f"{real_target_name}-bot: {redacted_response}")
                
                # Log both unredacted and redacted responses
                await self.log_response(user_name, real_target_name, content, unredacted_response, redacted_response)
            except discord.errors.Forbidden:
                print(f"Error: Bot doesn't have permission to send messages in {message.channel}")
            except Exception as e:
                print(f"An error occurred while processing the message: {e}")
        else:
            try:
                await message.channel.send("Please use the format: !username message")
            except discord.errors.Forbidden:
                print(f"Error: Bot doesn't have permission to send messages in {message.channel}")
            except Exception as e:
                print(f"An error occurred while sending the error message: {e}")

    async def generate_response(self, prompt: str, max_tokens: int = 200) -> tuple:
        if self.model is None:
            raise ValueError("Model has not been initialized")
        
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        eos_token_id = self.tokenizer.eos_token_id
        newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
        
        max_length = input_ids.shape[1] + max_tokens  # Limit total length to input + max_tokens
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_tokens):
            if generated_ids.shape[1] >= max_length:
                break
            
            with torch.no_grad():
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == eos_token_id or next_token_id.item() == newline_token_id:
                    break
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        # Split the response at the first newline and take only the first part
        response = response.split('\n')[0]
        
        return response, self.redact_text(response)

    def load_name_mapping(self, file_path: str):
        try:
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
                                    self.name_mapping[alias] = real_name
                        except:
                            return
            print("Name mapping loaded successfully")
        except Exception as e:
            print(f"Error loading name mapping: {e}")
            print("Continuing without name mapping")

    def load_filter_words(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        # Convert starred-out words to regex patterns
                        pattern = word.replace('*', r'\w*')
                        # Add word boundaries and make it case-insensitive
                        self.filter_patterns.append(re.compile(r'\b' + pattern + r'\b', re.IGNORECASE))
            print("Filter patterns loaded successfully")
        except Exception as e:
            print(f"Error loading filter patterns: {e}")
            print("Continuing without filter patterns")

    def redact_text(self, text: str) -> str:
        for pattern in self.filter_patterns:
            text = pattern.sub('[REDACTED]', text)
        return text
    
    async def log_response(self, user_name: str, target_name: str, content: str, unredacted_response: str, redacted_response: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {user_name} to {target_name}: {content}\n"
        log_entry += f"{target_name}-bot (Unredacted): {unredacted_response}\n"
        log_entry += f"{target_name}-bot (Redacted): {redacted_response}\n\n"
        with open("logs.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)
def load_api_key(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None
bot = InferenceBot()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Guild permissions: {bot.guilds[0].me.guild_permissions.value if bot.guilds else "Not in any guild"}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('!'):
        parts = message.content[1:].split(maxsplit=1)
        if len(parts) == 2:
            target_name, content = parts
            target_name = target_name.lower()
            
            if target_name in bot.name_mapping:
                real_target_name = bot.name_mapping[target_name]
            else:
                real_target_name = target_name.capitalize()
            
            user_name = bot.name_mapping.get(message.author.name.lower(), message.author.name)
            
            prompt = f"{user_name}: {content}\n{real_target_name}:"
            
            try:
                async with message.channel.typing():
                    unredacted_response, redacted_response = await bot.generate_response(prompt)
                
                # Send the message
                await message.channel.send(f"{real_target_name}-bot: {redacted_response}")
                
                # Log both unredacted and redacted responses
                await bot.log_response(user_name, real_target_name, content, unredacted_response, redacted_response)
            except discord.errors.Forbidden:
                print(f"Error: Bot doesn't have permission to send messages in {message.channel}")
            except Exception as e:
                print(f"An error occurred while processing the message: {e}")
        else:
            try:
                await message.channel.send("Please use the format: !username message")
            except discord.errors.Forbidden:
                print(f"Error: Bot doesn't have permission to send messages in {message.channel}")
            except Exception as e:
                print(f"An error occurred while sending the error message: {e}")

    await bot.process_commands(message)

async def main():
    base_model_path = "google/gemma-2b"  # Path to the base model
    lora_path = "./gemma_finetuned_lora2"  # Path to the LoRA adapters
    name_mapping_path = "name_mapping.txt"
    api_key_path = "key.txt"
    filter_words_path = "filter.txt"

    bot = InferenceBot()

    try:
        await bot.load_model_and_tokenizer(base_model_path, lora_path)
    except Exception as e:
        print(f"Failed to load model and tokenizer: {e}")
        return

    bot.load_name_mapping(name_mapping_path)
    bot.load_filter_words(filter_words_path)

    api_key = load_api_key(api_key_path)
    if api_key is None:
        print("Failed to load API key. Exiting.")
        return

    try:
        await bot.start(api_key)
    except discord.errors.LoginFailure:
        print("Failed to log in: Invalid token")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())