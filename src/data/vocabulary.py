from collections import Counter
from src.data.tokenizer import Tokenizer

class Vocabulary:
    def __init__(self, tokenizer_mode='basic'):
        # 1. Declare special tokens
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"  # Start of Sentence
        self.EOS_TOKEN = "<EOS>"  # End of Sentence
        self.UNK_TOKEN = "<UNK>"  # Unknown Word
        
        # 2. Standard index convention
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        
        # Two-way maps, Word to ID and ID to Word
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freqs = Counter()
        self.tokenizer = Tokenizer(mode=tokenizer_mode)
        
    def build_vocabulary(self, sentence_list, min_freq=2):
        """Build a vocabulary from a list of sentences."""
        for sentence in sentence_list:
            # Preprocess through Tokenizer
            for word in self.tokenizer.tokenize(sentence): 
                self.word_freqs[word] += 1
                
        # Keep words that appear at least min_freq times, removing noise and typos
        idx = 4
        for word, freq in self.word_freqs.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def numericalize(self, text):
        """Convert a string sentence into a list of index IDs."""
        tokenized_text = self.tokenizer.tokenize(text)
        return [self.word2idx.get(token, self.UNK_IDX) for token in tokenized_text]
        
    def __len__(self):
        return len(self.word2idx)
