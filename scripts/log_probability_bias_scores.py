import torch
import torch.nn.functional as F
#from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import pandas as pd
import numpy as np
import argparse
import copy
from tqdm import tqdm

####### CONFIG #######
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--demographic', type=str)
parser.add_argument('--template_file', type=str)
parser.add_argument('--attributes_file', type=str)
parser.add_argument('--out_file', type=str)
args = parser.parse_args()

BART_MODEL = args.model
DEMOGRAPHIC = args.demographic
TEMPLATE_FILE = args.template_file
ATTRIBUTES_FILE = args.attributes_file
OUT_FILE = args.out_file

####################################

# Load pre-trained model with masked language model head
tokenizer = BartTokenizer.from_pretrained(BART_MODEL)
model = BartForConditionalGeneration.from_pretrained(BART_MODEL)

# Load dataframe with attributes to permute through
attr_df = pd.read_csv(ATTRIBUTES_FILE, sep=',')
categories = np.unique(attr_df['category'])

# Demographic words to use to query and obtain probabilities for
all_tgt_words = {'GEND': {'male': ['man', 'he', 'male', 'm'],
                           'female': ['woman', 'she', 'female', 'f']},

                 'RACE': {'caucasian': ['caucasian', 'white'],
                         'asian': ['asian','chinese','korean','japanese','indian'],
                         'hispanic': ['hispanic','mexican'],
                         'african': ['african','black']},

                 'INSUR': {'medicare': ['medicare'],
                          'medicaid': ['medicaid'],
                          'private': ['private']},

                 'LANG': {'eng': ['english'],
                         'non-eng': ['russian','chinese','korean','spanish']}
                 }

TARGET_DICT = all_tgt_words[DEMOGRAPHIC]
FULL_DICT = [element for key in TARGET_DICT for element in TARGET_DICT[key]]

my_tgt_texts = []
my_prior_texts = []
my_categories = []

# clean up template sentences
templates = open(TEMPLATE_FILE).readlines()
templates = [x.rstrip('\n\r') for x in templates]
templates = [x.replace("[" + DEMOGRAPHIC + "]", '_') for x in templates]
templates = ["<s> " + x + " </s>" for x in templates]

# Generate target and prior sentences
for ATTRIBUTE in categories:
    for template in templates:
        if ATTRIBUTE in template:
            for words in attr_df.loc[attr_df['category'] == ATTRIBUTE, :].attribute:
                tmp = copy.deepcopy(template)

                tgt_text = tmp.replace("[" + ATTRIBUTE + "]", words)
                prior_text = tmp.replace("[" + ATTRIBUTE + "]", '_ ' * len(words.split(" ")))
                my_tgt_texts.append(tgt_text)
                my_prior_texts.append(prior_text)
                my_categories.append(ATTRIBUTE)

# Function for finding the target position (helper function for later)
def find_tgt_pos(text, tgt):
    txt = text.split(" ")
    for i in range(len(txt)):
        if tgt in txt[i]: # careful with things like "_," or "_."
            return i
    # if we've looped all positions but didn't find _
    print('Target position not found!')
    raise


# Return probability for the target word, and fill in the sentence (just for debugging)
def predict_word(text: str, model: BartForConditionalGeneration, tokenizer: BartTokenizer, tgt_word: str, tgt_pos: int):
    # print('Template sentence: ', text)
    mask_positions = []

    # insert mask tokens
    tokenized_text = tokenizer.tokenize(text)
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == 'Ġ_':
            tokenized_text[i] = '<mask>'
            mask_positions.append(i)

    #print("Here's tokenized_text:", tokenized_text)
    # Convert tokens to vocab indices
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    #print("Here's token_ids:", token_ids)
    tokens_tensor = torch.tensor([token_ids])

    # Call BART to calculate unnormalized probabilities for all pos
    model.eval()
    
    logits = model(tokens_tensor).logits
    
    # normalize by softmax
    predictions = F.softmax(logits, dim=2)

    # For the target word position, get probabilities for each word of interest
    # normalized = predictions[0, tgt_pos, :]
    # values, pred = normalized.topk(10)
    
    # out_prob = normalized[tokenizer.get_vocab()[tgt_word]].item()
    
    # Also, fill in all blanks by max prob, and print for inspection
    for mask_pos in mask_positions:
        out_preds = predictions[0, mask_pos, :]
        values, pred = out_preds.topk(10)
        found = False
        for tok in pred:
            word = tokenizer.decode(tok).lower()
            if word in FULL_DICT:
                found = True
                tokenized_text[mask_pos] = word
                break
        if found:
            continue
        else:
            predicted_index = torch.argmax(predictions[0, mask_pos, :]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            tokenized_text[mask_pos] = predicted_token

    for mask_pos in mask_positions:
        tokenized_text[mask_pos] = "_" + tokenized_text[mask_pos] + "_"
    pred_sent = ' '.join(tokenized_text).replace(' ##', '')

    return out_prob, pred_sent


# run through all generated templates and calculate results dataframe
results = {}
results['categories'] = []
results['demographic'] = []
results['tgt_text'] = []
results['log_probs'] = []
results['pred_sent'] = []

# Run through all generated permutations
for i in tqdm(range(len(my_tgt_texts))):
    tgt_text = my_tgt_texts[i]
    prior_text = my_prior_texts[i]
    print("Target texts:", tgt_text)
    print("Prior texts:", prior_text)

    #idx = 0
    for key, val in TARGET_DICT.items():
        # loop through the genders
        for tgt_word in val:
            tgt_pos = find_tgt_pos(tgt_text, '_')
            tgt_probs, pred_sent = predict_word(tgt_text, model, tokenizer, tgt_word, tgt_pos)
            prior_probs, _ = predict_word(prior_text, model, tokenizer, tgt_word, tgt_pos)

            # calculate log and store in results dictionary
            tgt_probs, pred_sent, prior_probs = np.array(tgt_probs), np.array(pred_sent), np.array(prior_probs)
            log_probs = np.log(tgt_probs / prior_probs)

            results['categories'].append(my_categories[i])
            results['demographic'].append(key)
            results['tgt_text'].append(my_tgt_texts[i])
            results['log_probs'].append(log_probs)
            results['pred_sent'].append(pred_sent)

# Write results to tsv
results = pd.DataFrame(results)
results.to_csv(OUT_FILE, sep='\t', index=False)
