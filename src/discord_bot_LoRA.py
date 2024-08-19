import discord
from discord.ext import commands

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, QuantoConfig

import asyncio
import datetime
import os
from aiocsv import AsyncWriter
import aiofiles
import pandas as pd
from typing import List, Dict

from utils import load_model_and_tokenizer, load_name_mapping, load_filter_words, load_auth_token, load_lora_adapter_from_base_model
from utils import async_redact_text, async_generate_response

os.environ["TOKENIZERS_PARALLELISM"] = "False"


# BASE_MODEL_PATH = "google/gemma-2-9b-it"  # Path to the base model
# LORA_ADAPTERS_PATH = "./gemma_finetuned_lora-9b"# Path to the LoRA adapters
BASE_MODEL_PATH = "HuggingFaceTB/SmolLM-135M-Instruct"
LORA_ADAPTERS_PATH = None
NAME_MAPPING_PATH = "name_mapping.txt"
DISCORD_BOT_AUTH_TOKEN_PATH = "discord_bot_auth_token.txt"
FILTER_WORDS_PATH = "filter.txt"
QUANTIZATION_CONFIG = QuantoConfig(weights="int4")


class InferenceBot(commands.Bot):
    
    # TODO change name mapping path to be optional
    # TODO add type hints
    def __init__(self, model_path, name_mapping_file_path, filter_words_file_path=None, quantization_config=None, lora_adapter_path=None):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents, permissions=discord.Permissions(3072))
        
        # TODO Make model and tokenizer and stuff have actual type hints
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, quantization_config)
        if lora_adapter_path:
            self.model = load_lora_adapter_from_base_model(self.model, lora_adapter_path)
        self.name_mapping = load_name_mapping(name_mapping_file_path)
        self.filter_patterns = load_filter_words(filter_words_file_path)
    
    async def on_ready(self):
        await self.download_guild_message_history(self.guilds[2])
        
        print(f'{self.user} has connected to Discord!')
        print(f'Guild permissions: {self.guilds[0].me.guild_permissions.value if self.guilds else "Not in any guild"}')
        

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        if message.content.startswith('!'):
            await self.handle_command(message)
        
        await self.process_commands(message)


    async def handle_command(self, message: discord.Message):
        parts = message.content[1:].split(maxsplit=1)
        if len(parts) == 2:
            target_name, content = parts
            target_name = target_name.lower()
            
            if target_name in self.name_mapping:
                real_target_name = self.name_mapping[target_name]
            else:
                real_target_name = target_name.capitalize()
            
            user_name = self.name_mapping.get(message.author.name.lower(), message.author.name)
            
            # Create the simple role-play instruction
            system_prompt = f"You are now role-playing as {real_target_name}. Respond as {real_target_name} would."
            # role_play_instruction = f"You are now role-playing as {real_target_name}. Respond as {real_target_name} would."

            prompt = f"{system_prompt}\n\n{user_name}: {content}\n{real_target_name}:"
            
            try:
                async with message.channel.typing():
                    unredacted_response = await async_generate_response(self.model, self.tokenizer, prompt)
                    redacted_response = await async_redact_text(unredacted_response, self.filter_patterns)
                
                # Split the response into chunks of 2000 characters or less
                chunks = [redacted_response[i:i+2000] for i in range(0, len(redacted_response), 2000)]
                
                # Send each chunk as a separate message
                for chunk in chunks:
                    await message.channel.send(f"{real_target_name}: {chunk}")
                
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

    
    async def log_response(self, user_name: str, target_name: str, content: str, unredacted_response: str, redacted_response: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {user_name} to {target_name}: {content}\n"
        log_entry += f"{target_name}-bot (Unredacted): {unredacted_response}\n"
        log_entry += f"{target_name}-bot (Redacted): {redacted_response}\n\n"
        with open("logs.txt", "a", encoding="utf-8") as log_file:
            log_file.write(log_entry)


    async def get_past_n_messages_in_channel(self, channel: discord.TextChannel, number_of_past_messages: int) -> List[discord.Message]:
        return [message async for message in channel.history(limit=number_of_past_messages)]

        
    
    
    # TODO if we need this to be fast in the future, keep track of a database of all of the messages in the server and the last time we updated the database, 
    # update the database with the new messages in the history that happen after the last updated time and then read the database for all of the user's messages
    # TODO potentially also shorten this method to just return a pandas dataframe or something and the user can decide whether or not they want to save it to a csv
    async def download_guild_message_history(self, guild: discord.Guild):
        channel_list: List[discord.TextChannel] = []
        for category in guild.categories:
            for channel in category.channels:
                if channel.permissions_for(guild.me).view_channel:
                    channel_list.append(self.get_channel(channel.id))
        
        guild_history_path = f"message_histories/{guild.name}/"
        
        # make sure that the directory exists
        dir_to_make = os.path.dirname(guild_history_path)
        if dir_to_make != "":
            await asyncio.to_thread(os.makedirs, name=dir_to_make, exist_ok=True)

        for channel in channel_list:
            async with aiofiles.open(f"{guild_history_path}/{channel.name}.csv", "w") as csv_file:
                csv_writer = AsyncWriter(csv_file)
                
                await csv_writer.writerow(["Author","Content"])
                    
                async for message in channel.history(limit=None):
                    if not message.content: continue
                    
                    content = message.content.replace("\n", " ")
                    
                    await csv_writer.writerow([message.author, content])




def main():

    # try:
    #     await bot.load_model_and_tokenizer(base_model_path, lora_path)
    # except Exception as e:
    #     print(f"Failed to load model and tokenizer: {e}")
    #     return

    # bot.load_name_mapping(name_mapping_path)
    # bot.load_filter_words(filter_words_path)
    
    bot = InferenceBot(
        model_path = BASE_MODEL_PATH, 
        name_mapping_file_path = NAME_MAPPING_PATH,
        filter_words_file_path = FILTER_WORDS_PATH,
        quantization_config = QUANTIZATION_CONFIG,
        lora_adapter_path = LORA_ADAPTERS_PATH
    )
    
    discord_bot_auth_token = load_auth_token(DISCORD_BOT_AUTH_TOKEN_PATH)
    if discord_bot_auth_token is None:
        print("Failed to load discord bot auth token. Exiting.")
        return

    asyncio.run(bot.start(discord_bot_auth_token))


if __name__ == "__main__":
    main()