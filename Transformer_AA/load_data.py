# -*- coding: utf-8 -*-
import torch
from torchtext.legacy import data
import re
#device = "cuda" if torch.cuda.is_available() else 'cpu'
device ='cpu'

def tokenizer(text):
    token = [tok for tok in list(text)]
    return token
    
def smiles_atom_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def data_gen(train_path='./train_data',train_data='uniprot_train_data.csv',valid_data='pdb_train_data.csv'):
    TEXT = data.Field(tokenize=tokenizer,
        init_token = '<sos>', 
        eos_token = '<eos>', 
        lower = False,  #True
        batch_first = True)

    train, val = data.TabularDataset.splits(
        path=train_path, 
        train=train_data,
        validation=valid_data,
        format='csv',
        skip_header=True,
        fields=[('trg', TEXT), ('src', TEXT)])

    TEXT.build_vocab(train, min_freq=2)
    id2vocab = TEXT.vocab.itos
    vocab2id = TEXT.vocab.stoi
    PAD_IDX = vocab2id[TEXT.pad_token]
    UNK_IDX = vocab2id[TEXT.unk_token]
    SOS_IDX = vocab2id[TEXT.init_token]
    EOS_IDX = vocab2id[TEXT.eos_token]

    #train_iter 自动shuffle, val_iter 按照sort_key排序，传入Decoder或者Encoder的sequence的长度不能超过模型中 position embedding 的 "vocabulary" size
    train_iter, val_iter = data.BucketIterator.splits(
        (train, val),
        batch_sizes=(8, 8),
        sort_key=lambda x: len(x.src),
        device=device)
    return train_iter, val_iter, id2vocab, PAD_IDX,TEXT,vocab2id,UNK_IDX

###################################################################################################################
def pre_data_gen(TEXT=None,train_path ='./train_data/uniprot_train_transformer' ,train_data='1_uniprot_train_transformer.csv',valid_data='56_uniprot_train_transformer.csv'):
    pre_train, pre_val = data.TabularDataset.splits(
        path=train_path, 
        train=train_data,
        validation=valid_data,
        format='csv',
        skip_header=True,
        fields=[('trg', TEXT), ('src', TEXT)])

    pre_train_iter, pre_val_iter = data.BucketIterator.splits(
        (pre_train, pre_val),
        batch_sizes=(8, 8),
        sort_key=lambda x: len(x.src),
        device=device)
    return pre_train_iter, pre_val_iter