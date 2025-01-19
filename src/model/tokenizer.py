import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}

    def _get_stats(self, corpus):
        """Get pair frequencies in the corpus."""
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, corpus):
        """Merge the most frequent pair in the vocabulary."""
        new_corpus = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(rf'(?<!\S){bigram}(?!\S)')
        for word, freq in corpus.items():
            new_word = pattern.sub(''.join(pair), word)
            new_corpus[new_word] = freq
        return new_corpus

    def train(self, corpus):
        """Train the BPE tokenizer on a given corpus."""
        # Initialize corpus with word frequencies
        word_freq = Counter(' '.join(list(word)) + ' </w>' for sentence in corpus for word in sentence.split())
        self.vocab = dict(word_freq)

        # Iteratively merge the most frequent pairs
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(self.vocab)
            if not pairs:
                break
            most_frequent = max(pairs, key=pairs.get)
            self.vocab = self._merge_vocab(most_frequent, self.vocab)
        return self.vocab