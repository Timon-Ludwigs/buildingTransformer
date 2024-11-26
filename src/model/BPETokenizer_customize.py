import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.base_vocab = None
    
    def preprocess_text(self, text):
        # Lowercase and split into words
        return text.lower().split()
    
    def get_stats(self, vocab):
        # Count frequency of pairs of characters/tokens
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, vocab, pair_to_merge):
        # Create a new vocabulary with the pair merged
        new_vocab = defaultdict(int)
        for word, freq in vocab.items():
            # Convert the word to a list of symbols
            new_word = word.replace(f"{pair_to_merge[0]} {pair_to_merge[1]}", 
                                     f"{pair_to_merge[0] + pair_to_merge[1]}")
            new_vocab[new_word] += freq
        return new_vocab
    
    def train(self, corpus):
        # Start with individual characters as tokens
        vocab = defaultdict(int)
        
        # Preprocess and count words
        for sentence in corpus:
            words = self.preprocess_text(sentence)
            for word in words:
                # Convert word to space-separated characters
                vocab[' '.join(list(word))] += 1
        
        # Keep track of merges
        self.merges = {}
        
        # Merge until we reach desired vocab size
        while len(self.vocab) < self.vocab_size:
            # Get pair frequencies
            pairs = self.get_stats(vocab)
            
            # If no more pairs, break
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the most frequent pair
            vocab = self.merge_vocab(vocab, best_pair)
            
            # Store the merge
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            
            # Update vocab
            self.vocab[best_pair[0] + best_pair[1]] = 1
        
        # Convert vocab to a set of tokens
        self.base_vocab = set(self.vocab.keys())
        return self
    
    def tokenize(self, text):
        # Preprocess text
        words = self.preprocess_text(text)
        
        # Tokenize each word
        tokens = []
        for word in words:
            # Start with characters
            word_tokens = list(word)
            
            # Apply merges
            while True:
                # Find best merge
                best_merge = None
                for merge_pair, merge_result in self.merges.items():
                    # Check if merge is possible
                    for i in range(len(word_tokens) - 1):
                        if (word_tokens[i], word_tokens[i+1]) == merge_pair:
                            best_merge = (i, merge_pair, merge_result)
                            break
                    if best_merge:
                        break
                
                # If no merge possible, break
                if not best_merge:
                    break
                
                # Perform merge
                i, _, merge_result = best_merge
                word_tokens = (word_tokens[:i] + 
                               [merge_result] + 
                               word_tokens[i+2:])
            
            tokens.extend(word_tokens)
        
        return tokens