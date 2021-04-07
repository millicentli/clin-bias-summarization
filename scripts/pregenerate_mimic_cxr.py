import torch
import argparse
import pickle
import numpy as np
import spacy
from pathlib import Path
from transformers import BartTokenizer

# Holds the data
class DocumentDatabase:
    def __init__(self):
        self.documents = []
        self.summaries = []
        self.dictionary = {}
        self.source_dictionary = {}
        self.target_dictionary = {}
        self.word_idx = 0
        self.src_idx = 0
        self.tgt_idx = 0
        self.doclen = 0

    def add_doc_sum(self, document, summary):
        self.documents.append(document)
        self.summaries.append(summary)
        if len(document) > self.doclen:
            self.doclen = len(document)
        #print("Here's document:", document)
        for word in document:
            #word = word.text.lower()
            word = word.lower()
            if word not in self.dictionary:
                self.dictionary[word] = self.word_idx
                self.word_idx += 1
                self.source_dictionary[word] = self.src_idx
                self.src_idx += 1
        for word in summary:
            #word = word.text.lower()
            word = word.lower()
            if word not in self.dictionary:
                self.dictionary[word] = self.word_idx
                self.word_idx += 1
                self.target_dictionary[word] = self.tgt_idx
                self.tgt_idx += 1

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds in self.documents:
            _dataset_sizes.append(len(ds.split()))
        return np.array(_dataset_sizes)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index], self.summaries[index]

# cleans the data
def clean_cxr(filename, database, tokenizer):
    sep = tokenizer.sep_token
    cls = tokenizer.cls_token
    with open(filename, 'r') as f:
        for line in f:
           split = line.split('\t')
           # Question: Do I need to do cls/sep for BART?
           # doc = cls + split[0] + sep
           # summary = cls + split[1] + sep
           doc = split[0]
           summary = split[1]
           database.add_doc_sum(doc, summary)

    return database

def apply_scispacy(filename, database, tokenizer):
    sep = tokenizer.sep_token
    cls = tokenizer.cls_token
    nlp = spacy.load('en_core_sci_md', disable=['tagger', 'ner'])
    with open(filename, 'r') as f:
        for line in f:
            split = line.split('\t')
            #print("Before spacy:", split[0])
            #doc = nlp(split[0].strip())
            #print("After spacy:", doc)
            #summary = nlp(split[1].strip())
            doc = split[0]
            summary = split[1]
            database.add_doc_sum(doc, summary)

    return database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--bart_model', type=str, required=True)

    args = parser.parse_args()

    tokenizer = BartTokenizer.from_pretrained(args.bart_model, do_lower_case=True)
    database = DocumentDatabase()
    database = apply_scispacy(args.train_df, database, tokenizer)

    # Save the database
    f = open(str(args.output_dir) + '/docs.pkl', 'wb')
    pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == '__main__':
    main()
    
