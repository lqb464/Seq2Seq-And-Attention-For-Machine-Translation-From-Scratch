import torch
import random
from src.data.dataset import TranslationDataset

class TranslationDataLoader:
    """
    Custom DataLoader implementation from scratch.
    Does not use torch.utils.data.DataLoader or torch.nn.utils.rnn.pad_sequence.
    
    Features:
    - Shuffle: randomly shuffle data order each epoch
    - Batching: split data into batches
    - Padding: pad <PAD> at the end of short sentences so all sentences in batch have the same length
    - Iterator: supports for-loop over each batch
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, pad_idx=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pad_idx = pad_idx
        
    def _pad_batch(self, sequences, pad_value):
        """
        Pad <PAD> at the end of short sentences in the batch.
        Equivalent to pad_sequence(batch_first=True) but handwritten.
        
        Args:
            sequences: list of 1D tensors with different lengths
            pad_value: value used for padding (usually PAD_IDX = 0)
            
        Returns:
            2D tensor shape (batch_size, max_seq_len)
        """
        # Find the maximum length in the batch
        max_len = max(len(seq) for seq in sequences)
        
        # Create a tensor filled with pad_value, then copy the real data into it
        padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
            
        return padded
        
    def __len__(self):
        """Return the number of batches (rounded up)."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """
        Create iterator to iterate over each batch.
        Each time __iter__ is called (each epoch), the data is shuffled again if needed.
        """
        # Create the index list
        indices = list(range(len(self.dataset)))
        
        # Shuffle: randomly mix the data order
        if self.shuffle:
            random.shuffle(indices)
        
        # Split into batches
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            
            # Get data from the dataset by index
            src_sequences = []
            tgt_sequences = []
            for idx in batch_indices:
                src, tgt = self.dataset[idx]
                src_sequences.append(src)
                tgt_sequences.append(tgt)
            
            # Pad all sentences in the batch to the same length
            src_padded = self._pad_batch(src_sequences, self.pad_idx)
            tgt_padded = self._pad_batch(tgt_sequences, self.pad_idx)
            
            yield src_padded, tgt_padded


def get_dataloader(src_sentences, tgt_sentences, src_vocab, tgt_vocab, batch_size=32, max_src_len_percentile=0.95, max_tgt_len_percentile=0.95, max_src_len=80, max_tgt_len=80):
    """Utility function: create a DataLoader from raw data."""
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_src_len_percentile, max_tgt_len_percentile, max_src_len, max_tgt_len)
    loader = TranslationDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pad_idx=src_vocab.PAD_IDX
    )
    return loader
