import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        """
        Initialize the dataset with data, tokenizer, and maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Store special token IDs
        self.special_tokens = {
            "pad": tokenizer.pad_token_id,
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
        }

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def process_sequence(self, token_ids, add_special_tokens=True):
        """
        Pad or truncate a list of token IDs to max_length.
        Add BOS/EOS tokens if specified.
        """
        if add_special_tokens:
            token_ids = [self.special_tokens["bos"]] + token_ids + [self.special_tokens["eos"]]
            effective_length = self.max_length - 2
        else:
            effective_length = self.max_length

        # Truncate the sequence if needed
        token_ids = token_ids[:effective_length]

        # Pad the sequence if needed
        padding_length = self.max_length - len(token_ids)
        token_ids += [self.special_tokens["pad"]] * padding_length

        return token_ids

    def create_attention_mask(self, token_ids):
        """
        Create an attention mask where 1 indicates a valid token and 0 indicates padding.
        """
        return [1 if token_id != self.special_tokens["pad"] else 0 for token_id in token_ids]

    def __getitem__(self, idx):
        """
        Fetch and process a single sample from the dataset.
        """
        sample = self.data[idx]
        
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample to be a dictionary, but got {type(sample).__name__}")

        source_text = sample["src"]
        target_text = sample["tgt"]

        # Encode source and target texts
        source_tokens = self.tokenizer.encode(source_text, add_special_tokens=False)
        target_tokens = self.tokenizer.encode(target_text, add_special_tokens=False)

        # Process source sequence
        source_ids = self.process_sequence(source_tokens, add_special_tokens=True)
        source_mask = self.create_attention_mask(source_ids)

        # Process target sequence
        target_input_ids = self.process_sequence(target_tokens, add_special_tokens=True)
        target_labels = target_input_ids[1:] + [self.special_tokens["pad"]]
        target_mask = self.create_attention_mask(target_input_ids)

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_input_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
            "labels": torch.tensor(target_labels, dtype=torch.long),
        }