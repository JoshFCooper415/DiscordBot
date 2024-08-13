import argparse
import json
from transformers import AutoTokenizer
import logging
from datetime import datetime

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def analyze_tokenizer_and_update_config(model_path, config_path, output_config_path, log_file):
    setup_logging(log_file)
    
    print(f"Loading tokenizer from {model_path}")
    logging.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading configuration from {config_path}")
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Analyzing tokenizer vocabulary for newline characters...")
    logging.info("Analyzing tokenizer vocabulary for newline characters...")
    newline_tokens = []
    for token, id in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([id])
        if '\n' in decoded or '\r' in decoded or '\*' in decoded:
            newline_tokens.append(id)
            logging.info(f"Newline token found - ID: {id}, Token: {token}, Decoded: {repr(decoded)}")
    
    print(f"Found {len(newline_tokens)} tokens containing newline characters.")
    logging.info(f"Found {len(newline_tokens)} tokens containing newline characters.")
    
    # Update the eos_token_id in the configuration
    if isinstance(config['eos_token_id'], list):
        config['eos_token_id'].extend(newline_tokens)
    else:
        config['eos_token_id'] = [config['eos_token_id']] + newline_tokens
    
    # Remove duplicates and sort
    config['eos_token_id'] = sorted(set(config['eos_token_id']))
    
    print(f"Updated eos_token_id: {config['eos_token_id']}")
    logging.info(f"Updated eos_token_id: {config['eos_token_id']}")
    
    # Save the updated configuration
    with open(output_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated configuration saved to {output_config_path}")
    logging.info(f"Updated configuration saved to {output_config_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a tokenizer for newline characters and update the configuration.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained model or tokenizer")
    parser.add_argument("config_path", type=str, help="Path to the input configuration file")
    parser.add_argument("output_config_path", type=str, help="Path to save the updated configuration file")
    parser.add_argument("--log_file", type=str, default="tokenizer_analysis.log", help="Path to the log file")
    args = parser.parse_args()

    analyze_tokenizer_and_update_config(args.model_path, args.config_path, args.output_config_path, args.log_file)

if __name__ == "__main__":
    main()