# -*- coding: utf-8 -*-
import torch
from load_data import  data_gen,pre_data_gen
pre_train_iter, pre_val_iter, id2vocab, PAD_IDX,TEXT,vocab2id,UNK_IDX = data_gen()
import pandas as pd
from model import Encoder, Decoder, Transformer
import re
device = "cuda" if torch.cuda.is_available() else 'cpu' 

#device ='cpu'

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

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device,max_length)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device,max_length)
model = Transformer(enc, dec, PAD_IDX, device).to(device)
model.load_state_dict(torch.load('final_finetuning_model.pt'))
model.eval()

#embeding_data = "./pretrain_data/AMP_standardseq_trans.csv"
embeding_data = "./final.txt"
embeding_data = pd.read_csv(embeding_data, sep=',',header=0)
#print(embeding_data)

def smiles_atom_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

for i in embeding_data["src"].values:
    tokens = smiles_atom_tokenizer(i)
    #tokens = [tok.lower() for tok in list(i)]

    tokens = [TEXT.init_token] + tokens + [TEXT.eos_token]

    src_indexes = [vocab2id.get(token, UNK_IDX) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    #print(i,enc_src)

    trg_indexes = [vocab2id[TEXT.init_token]]

    for i in range(1000):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)  
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    
        pred_token = output.argmax(2)[:,-1].item()   
        trg_indexes.append(pred_token)

        if pred_token == vocab2id[TEXT.eos_token]:
            trg_indexes = trg_indexes[:-1]
            break

    trg_tokens = [id2vocab[i] for i in trg_indexes]

    print("".join(trg_tokens[1:]))


'''
sent = '中新网9月19日电据英国媒体报道,当地时间19日,苏格兰公投结果出炉,55%选民投下反对票,对独立说“不”。在结果公布前,英国广播公司(BBC)预测,苏格兰选民以55%对45%投票反对独立。'
tokens = [tok for tok in jieba.cut(sent)]
tokens = [TEXT.init_token] + tokens + [TEXT.eos_token]
    
src_indexes = [vocab2id.get(token, UNK_IDX) for token in tokens]
src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
src_mask = model.make_src_mask(src_tensor)

with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)

trg_indexes = [vocab2id[TEXT.init_token]]

for i in range(50):
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    trg_mask = model.make_trg_mask(trg_tensor)  
    with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    
    pred_token = output.argmax(2)[:,-1].item()   
    trg_indexes.append(pred_token)

    if pred_token == vocab2id[TEXT.eos_token]:
        break

trg_tokens = [id2vocab[i] for i in trg_indexes]

print(trg_tokens[1:])
'''