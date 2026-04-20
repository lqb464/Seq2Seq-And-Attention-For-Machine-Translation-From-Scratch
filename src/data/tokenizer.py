import re
from collections import Counter

class Tokenizer:
    def __init__(self, mode='basic'):
        """
        Tokenizer supports 2 modes:
        - 'basic': Split words by spaces, lowercase text, and separate punctuation.
        - 'bpe': Byte-Pair Encoding, learns how to merge common characters into subwords.
        """
        self.mode = mode
        # BPE merge table, only used when mode='bpe'
        self.bpe_merges = []
        self.bpe_vocab = {}
        
    def tokenize(self, text):
        if self.mode == 'basic':
            return self._basic_tokenize(text)
        elif self.mode == 'bpe':
            return self._bpe_tokenize(text)
        else:
            raise ValueError(f"Unknown tokenizer mode: {self.mode}")
            
    def _basic_tokenize(self, text):
        """Preprocess text, convert to lowercase and split words by spaces."""
        text = text.lower()
        # Separate punctuation into individual tokens for model processing
        text = re.sub(r'([.!?,;:\'\"-])', r' \1 ', text)
        # Merge multiple consecutive spaces into one
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()

    # ========== BPE (Byte-Pair Encoding) ==========

    def learn_bpe(self, corpus, num_merges=1000):
        """
        Learn the BPE merge table from the corpus, which is a list of sentences.
        
        BPE algorithm:
        1. Split all words into character sequences and add the end-of-word token '</w>'.
        2. Count the frequency of each adjacent character pair.
        3. Merge the most frequent pair into one new token.
        4. Repeat steps 2 and 3 until num_merges is reached.
        """
        # Step 1: Count word frequency and split words into characters
        word_freqs = Counter()
        for sentence in corpus:
            for word in sentence.lower().split():
                # Each word = tuple of characters + end-of-word marker
                word_freqs[tuple(word) + ('</w>',)] += 1
        
        # Steps 2 to 4: Repeatedly merge the most frequent pair
        self.bpe_merges = []
        for i in range(num_merges):
            # Count the frequency of all adjacent pairs
            pairs = Counter()
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            if not pairs:
                break
                
            # Find the most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            self.bpe_merges.append(best_pair)
            
            # Merge that pair in all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        # Build vocabulary from the BPE result
        self.bpe_vocab = set()
        for word in word_freqs:
            for token in word:
                self.bpe_vocab.add(token)
    
    def _merge_pair(self, word, pair):
        """Merge one character pair in a word, represented as a tuple."""
        new_word = []
        i = 0
        while i < len(word):
            # If the target pair is found at position i
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def _apply_bpe(self, word):
        """Apply the learned BPE merge table to one word."""
        word = tuple(word) + ('</w>',)
        for pair in self.bpe_merges:
            word = self._merge_pair(word, pair)
        return list(word)
    
    def _bpe_tokenize(self, text):
        """
        Tokenize with BPE.
        If learn_bpe() has not been called, fall back to basic tokenization.
        """
        if not self.bpe_merges:
            return self._basic_tokenize(text)
        
        text = text.lower()
        words = text.split()
        tokens = []
        for word in words:
            subwords = self._apply_bpe(word)
            tokens.extend(subwords)
        return tokens