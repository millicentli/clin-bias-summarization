import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
from torch.utils import data
from utils import MIMICDataset, extract_embeddings, get_emb_size
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from run_classifier_dataset_utils import InputExample, convert_examples_to_features
from transformers import BartForConditionalGeneration, BartTokenizer
import Constants

parser = argparse.ArgumentParser('''Given a BART model and a dataset with a 'seqs' column, outputs a pickled dictionary
                                 mapping note_id to 2D numpy array, where each array is num_seq x emb_dim''')
parser.add_argument('--df_path', help = 'must have the following columns: seqs, num_seqs, and note_id either as a column or index')
parser.add_argument('--model_path', type = str)
parser.add_argument('--output_path', type = str)
parser.add_argument('--emb_method', default = 'last', const = 'last', nargs = '?', choices = ['last', 'sum4', 'cat4'], help = 'how to extract embeddings from BART output')
args = parser.parse_args()

df = pd.read_pickle(args.df_path)
if 'note_id' in df.columns:
    df = df.set_index('note_id')
tokenizer = BartTokenizer.from_pretrained(args.model_path)
model = BartForConditionalGeneration.from_pretrained(args.model_path, output_hidden_states=True)

def convert_input_example(note_id, text, seqIdx):
    return InputExample(guid = '%s-%s'%(note_id,seqIdx), text_a = text, text_b = None, label = 0, group = 0, other_fields = [])

examples = [convert_input_example(idx, i, c) for idx, row in df.iterrows() for c,i in enumerate(row.seqs)]
features = convert_examples_to_features(examples,
                                        Constants.MAX_SEQ_LEN, tokenizer, output_mode = 'classification')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

generator = data.DataLoader(MIMICDataset(features, 'train', 'classification'),  shuffle = True,  batch_size = n_gpu*32)

EMB_SIZE = get_emb_size(args.emb_method)
def get_embs(generator):
    model.eval()
    embs = {str(idx):np.zeros(shape = (row['num_seqs'], EMB_SIZE), dtype = np.float32) for idx, row in df.iterrows()}
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, _, _, guid, _ in tqdm(generator):
            input_ids = input_ids.to(device)
            # segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            #hidden_states, _ = model(input_ids, attention_mask = input_mask)
            output = model(input_ids, attention_mask=input_mask)
            # print("Here's the shape of output:", len(output))
            # not sure about this hidden state
            hidden_states_last = output[2]
            hidden_states = output[3]
            # print("Shape of hidden_states_last:", hidden_states_last.shape)
            #print("Shape of hidden_states_last:", hidden_states_last[-1].shape)
            #print("Shape of hidden_states[-1]:", hidden_states[-1].shape)
            #print("Shape of hidden_states[-2]:", hidden_states[-2].shape)
            #print("Shape of hidden_states[-3]:", hidden_states[-3].shape)
            #print("Shape of hidden_states[0]:", hidden_states[0].shape)

            #print("Here's the shape of last hidden:", hidden_states_last[:,0,:].shape)
            #print("Here's the shape of -1:", hidden_states[-1][:,0,:].shape)
            #print("Here's the shape of -2:", hidden_states[-2][:,0,:].shape)
            #print("Here's the shape of -3:", hidden_states[-3][:,0,:].shape)
            bart_out = extract_embeddings(hidden_states_last, hidden_states, args.emb_method)
           
            #print("Here's shape of bart_out:", bart_out.shape)
            for c,i in enumerate(guid):
                note_id, seq_id = i.split('-')
                emb = bart_out[c,:].detach().cpu().numpy()
                #print("Here's shape of emb:", emb.shape)
                #print("Checking the size required:", embs[note_id][int(seq_id), :].shape)
                embs[note_id][int(seq_id), :] = emb
    return embs

model_name = os.path.basename(os.path.normpath(args.model_path))
pickle.dump(get_embs(generator), open(args.output_path, 'wb'))
