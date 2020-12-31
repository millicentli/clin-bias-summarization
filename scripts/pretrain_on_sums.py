from argparse import ArgumentParser
from pathlib import Path
import torch
import logging
import pickle
import random
import collections
import json
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from functools import partial
from pregenerate_train_sums import DocumentDatabase
from dataset_util import DenoisingDataset, ConvertedDataset, collate

from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

'''
taken from https://github.com/pytorch/fairseq/blob/master/fairseq/data/encoders/utils.py with changes
'''
# Whole word masking (modified)
def get_whole_word_mask(dictionary):
    def is_beginning_of_word(i):
        tok = dictionary[i]
        if tok.startswith("madeupword"):
            return True
        try:
            if tok in ["<unk", "<s>", "</s>", "<pad>"]:
                return True
            else:
                return tok.startswith("\u0120")
        except ValueError:
            return True
    
    mask_whole_words = torch.ByteTensor(list(map(is_beginning_of_word, range(len(dictionary)))))
    return mask_whole_words

def load_vocab_file(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.read()
        vocab_dict = json.loads(tokens, object_pairs_hook=collections.OrderedDict)
    return vocab_dict
   
# General dataset
class PregeneratedDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        _input_ids = self.input_ids[index]
        _labels = self.labels[index]

        return _input_ids, _labels
 
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
    parser.add_argument('--learning_rate', default=3e-5, type=float, help="initial learning rate for Adam")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help="proportion of training to perform linear learning rate warmup for")
    parser.add_argument('--fp16', action='store_true', help="whether to use 16-bit float precision instead of 32-bit float precision")
    parser.add_argument('--loss_scale', type=float, default=0, help="loss scaling to improve fp16 numeric stability, used when fp16 is set to True\n"
                                                                    "0: dynamic loss scaling\n"
                                                                    "Power of 2: static loss scaling\n")
    # Dataset params
    parser.add_argument('--shuffle_instance', action='store_true', help='whether to shuffle the instances')
    parser.add_argument('--mask', type=float, default=0.3, help="ratio for masking")
    parser.add_argument('--mask_random', type=float, default=0.1, help="ratio for random masking")
    parser.add_argument('--insert', type=float, default=0.1, help="ratio for inserting")
    parser.add_argument('--rotate', type=float, default=0.1, help="ratio for rotating")
    parser.add_argument('--permute_sentences', type=float, default=1, help="ratio for permuting sentences")
    parser.add_argument('--replace_length', type=int, default=1, help="length of replacements")
    parser.add_argument('--mask_length', type=str, default='span-poisson', help="length of mask")
    parser.add_argument('--poisson_lambda', type=int, default=3, help="value of poisson")

    args = parser.parse_args()

    if not hasattr(args, "shuffle_instance"):
        args.shuffle_instance = False

    # preparing the model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.bart_model, do_lower_case=args.do_lower_case)
    model = BartForConditionalGeneration.from_pretrained(args.bart_model)

    # get the files and data
    f = open(args.pregenerated_data, "rb")
    print("Here's file:", f)
    database = pickle.load(f)
    print("Here's the length:", database.doclen)
    assert database is not None

    # TODO: set up the entire pipeline to encode the data here
    # Implement denoising of sentences
    # Call the function to get the data
    # Replace this with a URL
    vocab_to_idx = load_vocab_file('/gscratch/ark/limill01/clin-bias-summarization/bart/' + tokenizer.vocab_files_names['vocab_file'])
    idx_to_vocab = {idx : i for idx, i in enumerate(vocab_to_idx)}

    mask_whole_words = get_whole_word_mask(idx_to_vocab)
    dataset = PregeneratedDataset(database.documents, database.summaries)
    print("here's the dataset:", dataset)

    dataset = ConvertedDataset(dataset, tokenizer)
    
    mask_idx = vocab_to_idx['<mask>']
    bos_idx = vocab_to_idx['<s>']
    eos_idx = vocab_to_idx['</s>']
    pad_idx = vocab_to_idx['<pad>']

    print("here's the mask idx:", mask_idx)

    denoised_dataset = DenoisingDataset(
        dataset,
        database.sizes,
        idx_to_vocab,
        mask_idx,
        mask_whole_words,
        shuffle=args.shuffle_instance,
        seed=args.seed,
        args=args,
        eos=eos_idx,
        bos=bos_idx
    )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # calculate the total training examples (this could definitely change, so check this)
    num_train_optimization_steps = int(len(database) / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

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

    # preparing the optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_Scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
     
    num_warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    num_training_steps = int((1 - args.warmup_proportion) * num_train_optimization_steps)
    warmup_linear = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
   
    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(database)}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in range(args.epochs):
        epoch_dataset = denoised_dataset
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(
            epoch_dataset,
            sampler=train_sampler, 
            batch_size=args.train_batch_size,
            collate_fn=(lambda samples: collate(
                samples=samples,pad_idx=pad_idx, eos_idx=eos_idx, vocab=vocab_to_idx))
        )
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        # with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
        for step, batch in enumerate(epoch_iterator):
            input_ids = batch['net_input']['src_tokens'].to(device)
            decoder_input_ids = batch['net_input']['prev_output_tokens'].to(device)
            labels = batch['target'].to(device)
           
            '''
            print(type(batch))

            src_tokens = batch['net_input']['src_tokens'][0].numpy()
            print("checking:", src_tokens)
            string = [idx_to_vocab[s] for s in src_tokens]
            print("checking src 1:", string)  
            
            src_tokens2 = batch['net_input']['src_tokens'][1].numpy()
            print("checking 2:", src_tokens2)
            string = [idx_to_vocab[s] for s in src_tokens2]
            print("checking src 2:", string)

            print("SHAPE OF INPUT_IDS:", input_ids.size())
            
            prev_token = batch['net_input']['prev_output_tokens'][0].numpy()
            print("checking prev:", prev_token)
            string = [idx_to_vocab[s] for s in prev_token]
            print("checking prev token 1:", string)

            prev_token2 = batch['net_input']['prev_output_tokens'][1].numpy()
            print("checking prev 2:", prev_token2)
            string = [idx_to_vocab[s] for s in prev_token2]
            print("checking prev token 2:", string)

            print("SHAPE OF DECODER_IDS:", decoder_input_ids.size())

            target = batch['target'][0].numpy()
            print("checking target:", target)
            string = [idx_to_vocab[s] for s in target]
            print("checking target 1:", string)

            print("SHAPE OF LABELS:", labels.size())

            target2 = batch['target'][1].numpy()
            print("checking target:", target2)
            string = [idx_to_vocab[s] for s in target2]
            print("checking target 2:", string)
            '''

            outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
            loss = outputs[0]

            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
            epoch_iterator.set_postfix_str(f"Loss: {mean_loss:.5f}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
    
    # Saving a trained model
    logging.info("** ** * Saving finetuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model

    # Fix these two
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(args.output_dir, "config.json")

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    '''
    print(type(batch))

    src_tokens = batch['net_input']['src_tokens'][0].numpy()
    print("checking:", src_tokens)
    string = [idx_to_vocab[s] for s in src_tokens]
    print("checking src 1:", string)
    
    
    src_tokens2 = batch['net_input']['src_tokens'][1].numpy()
    print("checking 2:", src_tokens2)
    string = [idx_to_vocab[s] for s in src_tokens2]
    print("checking src 2:", string)
    
    
    prev_token = batch['net_input']['prev_output_tokens'][0].numpy()
    print("checking prev:", prev_token)
    string = [idx_to_vocab[s] for s in prev_token]
    print("checking prev token 1:", string)

    prev_token2 = batch['net_input']['prev_output_tokens'][1].numpy()
    print("checkign prev 2:", prev_token2)
    string = [idx_to_vocab[s] for s in prev_token2]
    print("checking prev token 2:", string)

    target = batch['target'][0].numpy()
    print("checking target:", target)
    string = [idx_to_vocab[s] for s in target]
    print("checking target 1:", string)

    target2 = batch['target'][1].numpy()
    print("checking target:", target2)
    string = [idx_to_vocab[s] for s in target2]
    print("checking target 2:", string)

    string = [idx_to_vocab[s] for s in batch['net_input']['prev_output_tokens']]
    print("checking output:", string)
    
    print("checking batch:", batch)
    batch = tuple(t.to(device) for t in batch)
    print("checking batch:", batch)
    '''
if __name__ == '__main__':
    main()

