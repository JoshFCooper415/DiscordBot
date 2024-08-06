import discord
from discord.ext import commands
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import asyncio
import re

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

    async def load_model_and_tokenizer(self, checkpoint_path: str, original_model_path: str):
        print(f"Loading tokenizer from {original_model_path}")
        print(f"Loading model from {checkpoint_path}")
       
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(original_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
            print("Model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")

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

    async def generate_response(self, prompt: str, max_length: int = 1000) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        eos_token_id = self.tokenizer.eos_token_id
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - len(input_ids[0])):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                if next_token_id.item() == eos_token_id or next_token_id.item() == self.tokenizer.encode('\n')[0]:
                    break
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

bot = InferenceBot()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Bot permissions: {bot.user.guild_permissions.value}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        # Extract all user mentions from the message
        mention_pattern = r'<@!?(\d+)>'
        mentions = re.findall(mention_pattern, message.content)
        
        # Remove the bot's ID from the mentions
        mentions = [m for m in mentions if int(m) != bot.user.id]
        
        if len(mentions) == 1:
            target_id = mentions[0]
            target_user = await bot.fetch_user(int(target_id))
            
            # Remove all mentions from the content
            content = re.sub(mention_pattern, '', message.content).strip()
            
            user_name = bot.name_mapping.get(message.author.name.lower(), message.author.name)
            target_name = bot.name_mapping.get(target_user.name.lower(), target_user.name)
            
            prompt = f"{user_name}: {content}\n{target_name}:"
            
            async with message.channel.typing():
                response = await bot.generate_response(prompt)
            
            await message.channel.send(f"{target_name}-bot: {response}")
        else:
            await message.channel.send("Please mention exactly one user to chat with.")

    await bot.process_commands(message)

def load_api_key(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error loading API key: {e}")
        return None

async def main():
    checkpoint_path = "models\chat_model_finetuned_final"
    original_model_path = "HuggingFaceTB/SmolLM-135M"
    name_mapping_path = "name_mapping.txt"
    api_key_path = "key.txt"

    await bot.load_model_and_tokenizer(checkpoint_path, original_model_path)
    bot.load_name_mapping(name_mapping_path)

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