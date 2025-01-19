import os
import sys
import json

# Add the root directory of the project to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset, config
# Print the cache directory path
print("Cache directory:", config.HF_DATASETS_CACHE)

#load_dataset.cleanup_cache_files
from src.utils.data_cleaning import clean_dataset
from src.model.tokenizer import BPETokenizer

def init_tokenizer(percentage):
    
    dataset = load_dataset("wmt17", "de-en", split=f"train[:{percentage}%]")
    #dataset = load_dataset("wmt17", "de-en", split=f"train[:10]")
    cleaned_data = clean_dataset(dataset)

    # Initialize and train the tokenizer
    tokenizer = BPETokenizer(vocab_size=50000)
    tokenizer.train(cleaned_data)
    #tokenizer.save("bpe_tokenizer.json")

# Initialize and train the tokenizer
init_tokenizer(10)
