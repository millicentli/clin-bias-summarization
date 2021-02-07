import math
import numpy as np
import torch
import contextlib
from torch.utils.data import Dataset, DataLoader

def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    tokenizer,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}
    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    def merge_mask(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return pad_mask(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    '''
    print("checking the samples inside of collate:", samples)
    print("checking number of samples:", len(samples))
    print("\n")
    print("printing the first sample:", samples[0])
    print("\n")
    print("printing the second sample", samples[1])
    '''

    # find the largest batch size for samples[x][0] - pad everything to that length
    # use the largest size for samples[x][1] to pad everything else with 0's
    # pad everything else with the pad value

    # aggregate the source tokens
    # :src_tokens = [s[0] for s in samples]
    checking = [s[0] for s in samples]
    #print("checking src tokens inside:", checking) 
    src_tokens = merge(
        0,
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # print("checking src_tokens", src_tokens)

    # sort by descending first sample length
    src_lengths = torch.LongTensor([s[0].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    src_tokens = src_tokens.index_select(0, sort_order)
    # print("here are the src_tokens order now:", src_tokens)
   
    source_mask = [s[1] for s in samples]
    labels = [s[2] for s in samples]
    # print("checking source_mask before:", source_mask)
    # print("checking y_id before:", y_id)
    # print("checking labels:", labels)
    # print("\n")
    # merge on the other tokens and also order them
    
    src_mask = merge(
        1,
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    
    labels = merge(
        2,
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    # Create the mask by checking the src labels
    for idx, (i, j) in enumerate(zip(src_tokens, src_mask)):
        # Find the location of the stop word
        # At that index in the mask, start iterating
        idx = (i  == eos_idx).nonzero().item() + 1
        j[idx:] = 0
        #print("Checking the new src_mask:", src_mask)


    # print("checking source_mask:", source_mask)
    # print("checking y_id:", y_id)
    # print("checking labels:", labels)

    return src_tokens, src_mask, labels

def pad_mask(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    size = max(v.size(1) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        dst = torch.unsqueeze(dst, 0)
        assert dst.numel() == src.numel()
        return dst.copy_(src)
    
    for i, v in enumerate(values):
        copy_tensor(v, res[i][: v.size(1)])

    exit()
    return res
    
def collate_tokens(
    values,
    pad_idx,
    eos_idx,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor"""
    size = max(v.size(1) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    res = values[0].new(len(values), size). fill_(pad_idx)

    def copy_tensor(src, dst):
        dst = torch.unsqueeze(dst, 0)
        # print("checking dst again:", dst.size())
        assert dst.numel() == src.numel()
        return dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][: v.size(1)])
    return res


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
        
        inputs = self.tokenizer.prepare_seq2seq_batch(ids.text, labels.text, truncation=True, padding=True, return_tensors='pt')
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

