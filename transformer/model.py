from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE
import os
from torch.utils.data import TensorDataset, DataLoader,Dataset
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from tqdm.notebook import tqdm
from sklearn.preprocessing import OrdinalEncoder
import random 

import gc
import pickle
import math
import datetime
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,Dataset
NUM_ITEMS = 30490

class ConvModule(nn.Module):
    def __init__(self, seq_len=56):
        super().__init__()
        self.seq_len = seq_len
        
        self.global_conv = nn.Conv1d(1, 2, kernel_size=(seq_len))
        self.week_conv = nn.Conv1d(1, 1, kernel_size=(7))
        self.last_week_conv = nn.Conv1d(1, 1, kernel_size=(7))
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x[:, None, :] #insert 1 channel (bs, channel, timesteps)
        bs = x.shape[0]
        
        out1 = self.global_conv(x).view(bs, -1).view(bs, -1)
        out1 = self.drop(out1)
        
        out2 = self.week_conv(x).view(bs, -1).view(bs, -1)
        out2 = self.drop(out2)
        
        out3 = self.last_week_conv(x[:, :, -7:]).view(bs, -1)
        out3 = self.drop(out3)     
        
        out = torch.cat([out1, out2, out3], axis=1)
        return out

uniques = [3049, 7, 10, 3, 3, 7, 31, 5, 5, 5]
dims = [16, 2, 3, 1, 1, 2, 4, 2, 2, 2]
emb_dims = [(x, y) for x, y in zip(uniques, dims)] #Hardcoding emb dims (not the best practice)

class M5Transformer(nn.Module):

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N_e: int,
                 N_d: int,
                 n_months: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: bool = True,
                 pe: str = None,
                 n_lag_convs=3,
                 e_days = 28*4,
                 d_days = 28):
        
        super().__init__()
        
        self._d_model = d_model
        self._inp_dropout = nn.Dropout(0.1)

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self._lag_per_time = nn.Linear(63, n_lag_convs)
        
        self._e_inp_norm = nn.LayerNorm(d_input+n_lag_convs)
        self._d_inp_norm = nn.LayerNorm(d_input+n_lag_convs-1)

        self._embedding = nn.Linear(d_input, d_model)
        
        self.enc_embedding = nn.Linear(d_input+n_lag_convs, d_model)
        self.dec_embedding = nn.Linear(d_input-1+n_lag_convs, d_model)
        
        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N_e)])
        
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      n_months,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N_d)])

        self.hist_covs = ConvModule(e_days)
        out_dim = 237
        self.out = nn.Sequential(
                                        nn.Linear(out_dim, out_dim//2),
                                        nn.LeakyReLU(),
                                        nn.LayerNorm(out_dim//2),
                                        nn.Linear(out_dim//2, d_output),
                                    )
#         self.out = nn.Linear(out_dim, d_output)
        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

#     def lag_features(self, lag):
#         bs, t, nf = lag.shape
#         lag_features = []        
#         for i in range(t):
#                 _x = lag[:, i, :].reshape(bs, nf, -1)
#                 feats = [lc(_x) for lc in self._lag_convs]
#                 lag_features.append(torch.cat(feats, dim=2))
#         return torch.cat(lag_features, dim=1)
                
    def embed_and_combine(self, cont, cat, lag):
        cat = cat.type(torch.LongTensor).to(device)
        cont = cont.type(torch.FloatTensor).to(device)
        lag = lag.type(torch.FloatTensor).to(device)                
        xcat = [el(cat[:, :, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, axis=2)
        xlags = self._lag_per_time(lag)        
        combined = torch.cat([xcat, cont, xlags], axis=2) 
        combined = self._inp_dropout(combined)  
        return combined 

                
    def forward(self, x):
        hist, e_cont, e_cat, e_lag, d_cont, d_cat, d_lag = x        
                
        encoder_input = self.embed_and_combine(e_cont, e_cat, e_lag) #(bs, etime, d_inp)
        encoder_input = self._e_inp_norm(encoder_input)
        decoder_input = self.embed_and_combine(d_cont, d_cat, d_lag) #(bs, etime, d_inp-1)
        decoder_input = self._d_inp_norm(decoder_input)
                
        bs = encoder_input.shape[0]         
        Ke = encoder_input.shape[1]
        Kd = decoder_input.shape[1]

        # Embedding module
        encoding = self.enc_embedding(encoder_input)        

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(Ke, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = self.dec_embedding(decoder_input)        
#         decoding = encoding        

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(Kd, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            decoding.add_(positional_encoding)
                
        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        _hist = self.hist_covs(hist.type(torch.FloatTensor).to(device))       
        _hist = _hist.unsqueeze(1).repeat(1, 28, 1)        
        output = self.out(torch.cat([decoding, _hist], 2))
        output = F.relu(output) #Demand can only be +ve
        return output.reshape(bs, -1)