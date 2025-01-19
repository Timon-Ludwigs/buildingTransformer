import os
import sys

# Add the root directory of the project to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.tokenizer import BPETokenizer

corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]

# Train the custom BPE tokenizer
bpe_tokenizer = BPETokenizer(vocab_size=64)
vocab = bpe_tokenizer.train(corpus)
print(f"Trained Vocabulary: {vocab}")
