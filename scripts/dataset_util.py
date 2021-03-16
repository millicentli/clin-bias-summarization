import math
import numpy as np
import torch
import contextlib
from torch.utils.data import Dataset, DataLoader

def collate_fn(
    samples, 
    tokenizer
):
    src, labels = zip(*samples)
    src_out = tokenizer(src, padding=True, return_tensors='pt', max_length=117)
    
    src_tokens = src_out['input_ids']
    atten = src_out['attention_mask']

    labels_tok_list = []
    maxlen = 0
    for label in labels:
        tokens = tokenizer.encode(label)
        maxlen = max(maxlen, len(tokens))
        labels_tok_list.append(tokens)

    n = len(labels_tok_list)
    d = maxlen
    labels_arr = np.zeros((n, d), dtype=int)

    for idx, label in enumerate(labels_tok_list):
        labels_arr[idx][:len(label)] = label
        labels_arr[idx][len(label):] = 1

    labels = torch.LongTensor(labels_arr)
    
    return src_tokens, atten, labels

# Data converted for fine-tuning
class ConvertedDataset(Dataset):
    """
    Dataset to convert words to appropriate indices
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        pad_idx
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.pad_idx = pad_idx
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # gets the input and label
        ids, labels = self.dataset[index]
        
        inputs = self.tokenizer.prepare_seq2seq_batch(ids, labels, truncation=True, padding=True, return_tensors='pt')
        source_ids, source_mask, y = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]

        return source_ids, source_mask, y

# General dataset
class PregeneratedDataset(Dataset):
    """
    Dataset to hold the general documents and summaries
    """

    def __init__(self, 
        input_ids, 
        labels
    ):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        _input_ids = self.input_ids[index]
        _labels = self.labels[index]

        return _input_ids, _labels

