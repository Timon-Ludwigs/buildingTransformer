import re
import unicodedata

def clean_text(text, whitelist="abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\|_+*¥"):
    """
    Clean text by removing non-whitelisted characters and normalizing unicode.
    
    Args:
        text (str): Input text to clean
        whitelist (str): Allowed characters
    
    Returns:
        str: Cleaned text
    """
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove non-whitelisted characters
    text = ''.join(char for char in text if char in whitelist)
    
    # Convert to lowercase
    text = text.lower().strip()
    
    return text

def clean_dataset(dataset, min_length=5, max_length=64, max_ratio=1.5):
    """
    Clean the machine translation dataset.
    
    Args:
        dataset: Huggingface dataset
        min_length (int): Minimum sentence length
        max_length (int): Maximum sentence length
        max_ratio (float): Maximum length ratio between source and target
    
    Returns:
        Cleaned dataset
    """
    def is_valid_pair(source, target):
        # Clean texts
        source_clean = clean_text(source)
        target_clean = clean_text(target)
        
        # Check lengths
        if not (min_length <= len(source_clean.split()) <= max_length and 
                min_length <= len(target_clean.split()) <= max_length):
            return False
        
        # Check length ratio
        if len(source_clean) / len(target_clean) > max_ratio or \
           len(target_clean) / len(source_clean) > max_ratio:
            return False
        
        return True
    
    # Filter dataset
    cleaned_dataset = dataset.filter(
        lambda x: is_valid_pair(x['translation']['de'], x['translation']['en'])
    )
    
    return cleaned_dataset