# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import Encoder, Decoder, Transformer
import sys
device = "cuda" if torch.cuda.is_available() else 'cpu' 
#device ='cpu'

print("device:",device)

v2 = int(sys.argv[1])
model_file = sys.argv[2]
train_data = sys.argv[3]

from load_data import  data_gen,pre_data_gen
pre_train_iter, pre_val_iter, id2vocab, PAD_IDX,TEXT,vocab2id,UNK_IDX = data_gen()

if v2==1:
    #'1_uniprot_train_transformer.csv'
    pre_train_iter, pre_val_iter = pre_data_gen(TEXT=TEXT,train_path ='./train_data/uniprot_train_transformer' ,
    train_data=train_data,valid_data='56_uniprot_train_transformer.csv')
else:
    #'0_pdb_train_transformer.csv'
    pre_train_iter, pre_val_iter = pre_data_gen(TEXT=TEXT,train_path ='./pretrain_data/pdb_train_transformer' ,
    train_data=train_data,valid_data='18_pdb_train_transformer.csv')

INPUT_DIM = len(id2vocab)
OUTPUT_DIM = len(id2vocab)
HID_DIM = 256
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 2
DEC_HEADS = 2
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
N_EPOCHS = 100
CLIP = 1
max_length = 2000

print(INPUT_DIM)

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device,max_length)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device,max_length)
model = Transformer(enc, dec, PAD_IDX, device).to(device)
model.load_state_dict(torch.load(model_file))

save_pt_freq = 10

optimizer = optim.Adam(model.parameters(), lr=5e-5)
#we ignore the loss whenever the target token is a padding token.
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

loss_vals = []
loss_vals_eval = []
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(pre_train_iter)
    pbar.set_description("[finetuning_Train Epoch {}]".format(epoch)) 
    for batch in pbar:
        trg, src = batch.trg.to(device), batch.src.to(device)
        model.zero_grad()
        output, _ = model(src, trg[:,:-1])
        #trg = [batch size, trg len]
        #output = [batch size, trg len-1, output dim]        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)                     
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]     
        loss = criterion(output, trg)    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        epoch_loss.append(loss.item())
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss))
    
    model.eval()
    epoch_loss_eval= []
    pbar = tqdm(pre_val_iter)
    pbar.set_description("[finetuning_Eval Epoch {}]".format(epoch)) 
    for batch in pbar:
        trg, src = batch.trg.to(device), batch.src.to(device)
        model.zero_grad()
        output, _ = model(src, trg[:,:-1])
        #trg = [batch size, trg len]
        #output = [batch size, trg len-1, output dim]        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)                     
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]   
        loss = criterion(output, trg)    
        epoch_loss_eval.append(loss.item())
        pbar.set_postfix(loss=loss.item())
    loss_vals_eval.append(np.mean(epoch_loss_eval))  

    if (epoch+1)%save_pt_freq ==0:
        torch.save(model.state_dict(), str(epoch+1)+'_finetuning_model.pt') 
        print("save model:",str(epoch+1)+'_finetuning_model.pt')
    
torch.save(model.state_dict(), 'final_finetuning_model.pt')

print(loss_vals,loss_vals_eval)