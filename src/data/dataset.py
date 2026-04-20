import torch

class TranslationDataset:
    """
    Dataset for machine translation task.
    Written from scratch, does not inherit from torch.utils.data.Dataset.
    Stores list of source/target sentences and converts text to tensor when accessed.
    """
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_src_len_percentile=0.95, max_tgt_len_percentile=0.95, max_src_len=80, max_tgt_len=80):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len_percentile = max_src_len_percentile
        self.max_tgt_len_percentile = max_tgt_len_percentile
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Calculate percentile lengths if needed
        if isinstance(self.max_src_len_percentile, float) and 0 < self.max_src_len_percentile < 1:
            src_lengths = [len(self.src_vocab.numericalize(s)) for s in src_sentences]
            src_lengths.sort()
            self.src_limit = max(1, int(src_lengths[int(len(src_lengths) * self.max_src_len_percentile)]))
        else:
            self.src_limit = self.max_src_len
            
        if isinstance(self.max_tgt_len_percentile, float) and 0 < self.max_tgt_len_percentile < 1:
            tgt_lengths = [len(self.tgt_vocab.numericalize(s)) for s in tgt_sentences]
            tgt_lengths.sort()
            self.tgt_limit = max(1, int(tgt_lengths[int(len(tgt_lengths) * self.max_tgt_len_percentile)]))
        else:
            self.tgt_limit = self.max_tgt_len
        
    def __len__(self):
        return len(self.src_sentences)
        
    def __getitem__(self, index):
        src_text = self.src_sentences[index]
        tgt_text = self.tgt_sentences[index]
        
        # Convert string to list of indices, add SOS at start and EOS at end
        num_src = [self.src_vocab.SOS_IDX] + self.src_vocab.numericalize(src_text) + [self.src_vocab.EOS_IDX]
        num_tgt = [self.tgt_vocab.SOS_IDX] + self.tgt_vocab.numericalize(tgt_text) + [self.tgt_vocab.EOS_IDX]
        
        # Trim according to percentile limit
        num_src = num_src[:self.src_limit]
        num_tgt = num_tgt[:self.tgt_limit]
        
        return torch.tensor(num_src), torch.tensor(num_tgt)
