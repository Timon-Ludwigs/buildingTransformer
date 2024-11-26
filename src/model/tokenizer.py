# For comparison with Hugging Face implementation
import tokenizers
from BPETokenizer_customize import BPETokenizer

def train_huggingface_tokenizer(corpus, vocab_size=295):
    # Create a new BPE tokenizer
    tokenizer = tokenizers.BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )
    
    # Train the tokenizer
    tokenizer.train_from_iterator(
        corpus,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True
    )
    
    return tokenizer

# Corpus for training
corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]

# Train custom BPE tokenizer
custom_tokenizer = BPETokenizer(vocab_size=64).train(corpus)

# Example sentence to tokenize
test_sentence = "Machine learning is a subset of artificial intelligence."

# Tokenize with custom implementation
custom_tokens = custom_tokenizer.tokenize(test_sentence)
print("Custom Tokenizer Tokens:", custom_tokens)

# Train Hugging Face tokenizer
hf_tokenizer = train_huggingface_tokenizer(corpus)

# Tokenize with Hugging Face implementation
hf_tokens = hf_tokenizer.encode(test_sentence).tokens
print("Hugging Face Tokenizer Tokens:", hf_tokens)