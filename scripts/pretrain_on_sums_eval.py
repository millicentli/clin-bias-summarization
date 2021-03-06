from argparse import ArgumentParser
from pathlib import Path
import torch
import logging
import pickle
import random
import collections
import json
import json
import os
import csv
import shutil
import numpy as np
import pandas as pd

import pdb

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from functools import partial
from pregenerate_train_sums import DocumentDatabase
from dataset_util import ConvertedDataset, PregeneratedDataset, collate_fn
from rouge import Rouge

from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup

### TODO: Clean up extraneous code that's not necessary

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
writer = SummaryWriter()

# Quick evaluation
def evaluate(args, model, tokenizer, data, device):
    logging.info("***** Running evaluation *****")

    total = 0
    nb_eval_step = 0

    predictions = []
    actual = []
    
    dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=(lambda samples: collate_fn(
            samples=samples, tokenizer=tokenizer))
    )

    rouge = Rouge()

    model.to(device)
    model.eval()
    for batch in tqdm(dataloader):
        inputs, atten, labels = batch
        inputs = inputs.to(device)
        atten = atten.to(device)
        labels = labels.to(device)
        
        # Predicted words
        out_ypred = model.generate(inputs, max_length=100, num_beams=5, length_penalty=2.0, early_stopping=True)
        words_ypred = tokenizer.decode(out_ypred[0].tolist(), skip_special_tokens=True)

        # Actual words
        words_label = tokenizer.decode(labels[0].tolist(), skip_special_tokens=True)


        predictions.append(words_ypred)
        actual.append(words_label)

    scores = rouge.get_scores(predictions, actual, avg=True)

    # Writing this to CSV
    with open('bart_rouge_results.csv', 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter=',')
        csvwriter.writerow(['type', 'f', 'p', 'r'])
        for score in scores:
            csvwriter.writerow([score, scores[score]['f'], scores[score]['p'], scores[score]['r']]) 
    

# Quick validation
def validate(args, model, tokenizer, data, device, epoch, model_losses):
    logging.info("***** Running validation *****")

    dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=(lambda samples: collate_fn(
            samples=samples, tokenizer=tokenizer))
    )     

    dev_loss = 0.0
    nb_dev_step = 0
    
    model.eval()
    for batch in tqdm(dataloader, desc="Checking dev model accuracy..."):
        with torch.no_grad():
            inputs, atten, labels = batch
            inputs = inputs.to(device)
            atten = atten.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=inputs, attention_mask=atten, labels=labels)
            tmp_dev_loss, _ = outputs[:2]

            dev_loss += tmp_dev_loss.item()
            nb_dev_step += 1

    loss = dev_loss / nb_dev_step
    model_losses.append(loss)
    writer.add_scalar('dev_loss', loss, epoch)

    # Saving a trained model
    logging.info("** ** * Saving validated model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model

    path = str(epoch)
    full_path = os.path.join(args.output_dir, path)
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    os.mkdir(full_path)
    # os.mkdir(os.path.join(args.output_dir, path))
    output_model_file = os.path.join(args.output_dir, path, "pytorch_model.bin")
    output_config_file = os.path.join(args.output_dir, path, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, path))

class Dataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        _id = self.ids[index]
        _mask = self.masks[index]
        _label = self.labels[index]

        return _id, _mask, _label

def main():
    parser = ArgumentParser()
    # General params
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--bart_model', type=str, required=True, help="choose a BART pre-trained model: facebook/bart-base, facebook/bart-large, facebook/bart-large-mnli")
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--epochs', type=int, default=3, help="number of epochs to train for")
    parser.add_argument('--local_rank', type=int, default=-1, help="for distributed training on gpus")
    parser.add_argument('--no_cuda', action='store_true', help="whether or not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument('--train_batch_size', default=32, type=int, help="total batch size for training")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="initial learning rate for Adam")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help="proportion of training to perform linear learning rate warmup for")
    parser.add_argument('--fp16', action='store_true', help="whether to use 16-bit float precision instead of 32-bit float precision")
    parser.add_argument('--loss_scale', type=float, default=0, help="loss scaling to improve fp16 numeric stability, used when fp16 is set to True\n"
                                                                    "0: dynamic loss scaling\n"
                                                                    "Power of 2: static loss scaling\n")
    parser.add_argument('--grad_clip', type=float, default=0.25, help="Grad clipping value")
    args = parser.parse_args()

    tokenizer = BartTokenizer.from_pretrained(args.bart_model)
    model = BartForConditionalGeneration.from_pretrained(args.bart_model)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    
    # get files and data
    f = open(args.pregenerated_data, "rb")
    database = pickle.load(f)
    assert database is not None
    
    # vocab
    vocab_to_idx = tokenizer.get_vocab()
    idx_to_vocab = {idx : i for idx, i in enumerate(vocab_to_idx)}

    # special tokens
    mask_idx = vocab_to_idx['<mask>']
    bos_idx = vocab_to_idx['<s>']
    eos_idx = vocab_to_idx['</s>']
    pad_idx = vocab_to_idx['<pad>']

    # Datasets
    
    train_idx = int(0.8 * len(database.documents))
    test_idx = int(0.1 * len(database.documents))
    dataset_train = PregeneratedDataset(database.documents[:train_idx], database.summaries[:train_idx])
    dataset_dev = PregeneratedDataset(database.documents[train_idx: len(database.documents) - test_idx], database.summaries[train_idx: len(database.documents) - test_idx])
    dataset_test = PregeneratedDataset(database.documents[train_idx + test_idx:], database.summaries[train_idx + test_idx:])
    

    device = None
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if not hasattr(args, "shuffle_instance"):
        args.shuffle_instance = False

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    

    model.to(device)
    params = [p for n,p in model.named_parameters()]
    optimizer = AdamW(params, lr=args.learning_rate)

    global_step = 0
    logging.info("****** Running training *****")
    logging.info(f"  Num examples = {len(dataset_train)}")
    logging.info(f"  Num epochs = {args.epochs}")
    logging.info(f"  Batch size = {args.train_batch_size}")
    logging.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    
    model_losses = []
    total_loss = 0.0
    
    train_iterator = trange(0, args.epochs, desc="Epoch")
    for epoch, _ in enumerate(train_iterator):
        
        epoch_dataset = dataset_train
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        
        train_dataloader = DataLoader(
            epoch_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=(lambda samples: collate_fn(
                                samples=samples, tokenizer=tokenizer))
        )

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        for step, batch in enumerate(epoch_iterator):
            model.train()

            input_ids, atten, labels = batch

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=atten, labels=labels)
            loss = outputs[0]

            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.zero_grad()
            global_step += 1
            epoch_iterator.set_description(f"Loss: {total_loss / (global_step + 1)}")
        
        
        validate(args, model, tokenizer, dataset_dev, device, epoch, model_losses)
   
    writer.flush()
    writer.close()

    loss_arr = np.asarray(model_losses)

    df = pd.DataFrame(loss_arr)

    # Writing the output of the validation
    df.to_csv("bart_validation_results.csv", index=False) 
    
    # Evaluate - use the best model
    min_loss_idx = np.argmin(loss_arr)
    
    path = str(min_loss_idx)
    tokenizer = BartTokenizer.from_pretrained(os.path.join(args.output_dir, path))
    model = BartForConditionalGeneration.from_pretrained(os.path.join(args.output_dir, path))
    
    evaluate(args, model, tokenizer, dataset_test, device)

if __name__ == '__main__':
    main()
