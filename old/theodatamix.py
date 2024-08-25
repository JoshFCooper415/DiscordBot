import re
import os
from datasets import load_dataset, Dataset
from multiprocessing import Pool, cpu_count, shared_memory
import numpy as np
from tqdm import tqdm
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore")

def remove_links(text):
    return re.sub(r'http\S+', '', text, flags=re.IGNORECASE)

def process_chunk(args):
    try:
        chunk_id, start, end, text_key, title_key, shm_name = args
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        chunk = np.frombuffer(existing_shm.buf[start:end], dtype=np.uint8).tobytes().decode('utf-8')
        samples = eval(chunk)
        
        texts = []
        for sample in samples:
            try:
                text = sample.get(text_key)
                if not isinstance(text, str):
                    continue
                
                text = remove_links(text)
                
                if title_key:
                    title = sample.get(title_key)
                    if not isinstance(title, str):
                        continue
                    title = remove_links(title)
                    text = f"{title}: {text}"
                
                # Removed the filter_regex check
                
                if len(text.split()) >= 500:
                    continue
                
                texts.append(text)
            except Exception as e:
                logging.error(f"Error processing sample: {str(e)}")
                continue
        
        return texts
    except Exception as e:
        logging.error(f"Error in process_chunk: {str(e)}")
        return []

def process_dataset(dataset, text_key, title_key=None, dataset_name=None):
    try:
        batch_size = 10000
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        
        serialized_data = str(dataset[:]).encode('utf-8')
        shm = shared_memory.SharedMemory(create=True, size=len(serialized_data))
        np.frombuffer(shm.buf, dtype=np.uint8)[:] = np.frombuffer(serialized_data, dtype=np.uint8)
        
        process_args = [
            (i, i*batch_size, min((i+1)*batch_size, len(dataset)), text_key, title_key, shm.name)
            for i in range(num_batches)
        ]
        
        with Pool(cpu_count()) as pool:
            for processed_chunk in tqdm(pool.imap(process_chunk, process_args), total=num_batches, desc=f"Processing {dataset_name}", leave=True):
                yield from processed_chunk
        
        shm.close()
        shm.unlink()
    except Exception as e:
        logging.error(f"Error in process_dataset: {str(e)}")
        yield from []

datasets_to_process = [
    ("fschlatt/trump-tweets", 'text', None),
    ("argilla/twitter-coronavirus", 'text', None),
    ("zachgitt/comedy-transcripts", 'transcript', None),
    ("sentence-transformers/reddit", 'body', 'title')
]

try:
    all_texts = []

    for dataset_name, text_key, title_key in datasets_to_process:
        try:
            logging.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split='train')
            logging.info(f"Dataset {dataset_name} loaded. Size: {len(dataset)}")
            
            processed_texts = list(process_dataset(dataset, text_key, title_key, dataset_name))
            logging.info(f"Processed {len(processed_texts)} texts from {dataset_name}")
            
            all_texts.extend(processed_texts)
            logging.info(f"Total texts collected so far: {len(all_texts)}")
                
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {str(e)}")
            continue

    logging.info(f"Total texts collected from all datasets: {len(all_texts)}")

    # Create a new dataset from all processed texts
    final_dataset = Dataset.from_dict({"text": all_texts})

    # Check if the dataset is not empty
    if len(final_dataset) == 0:
        raise ValueError("The final dataset is empty. No data to save.")

    # Save the dataset
    save_path = "custom_datamix"
    logging.info(f"Saving dataset to {save_path}")
    final_dataset.save_to_disk(save_path)

    # Verify that the dataset was saved
    if not os.path.exists(save_path):
        raise IOError(f"Failed to save the dataset. The directory {save_path} does not exist.")

except Exception as e:
    logging.error(f"A critical error occurred: {str(e)}")
    import traceback
    logging.error(traceback.format_exc())
else:
    logging.info(f"Processing complete. Total rows in combined dataset: {len(final_dataset)}")
    logging.info(f"Dataset saved to 'custom_datamix' directory")

# Add this at the end to check the content of the saved dataset
try:
    loaded_dataset = Dataset.load_from_disk("custom_datamix")
    logging.info(f"Loaded saved dataset. Size: {len(loaded_dataset)}")
    if len(loaded_dataset) > 0:
        logging.info(f"First item in saved dataset: {loaded_dataset[0]}")
    else:
        logging.warning("Saved dataset is empty")
except Exception as e:
    logging.error(f"Error loading saved dataset: {str(e)}")