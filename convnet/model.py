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

#Hard coded embedding dimensions for now
uniques = [NUM_ITEMS, 7, 10, 3, 3, 7, 12, 6, 31, 5, 5, 5]
# dims = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dims = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
emb_dims = [(x, y) for x, y in zip(uniques, dims)]
print(emb_dims)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class LinearBlock(nn.Module):
    def __init__(self, in_d, out_d, p=0):
        super().__init__()
        self.block = nn.Sequential(
                            nn.Linear(in_d, out_d),
                            nn.ReLU(),
#                             nn.LeakyReLU(),
#                             nn.Tanh(),
        )
    
    def forward(self, x):
        return self.block(x)
    
class ConvModule(nn.Module):
    def __init__(self, seq_len=56):
        super().__init__()
        self.seq_len = seq_len
        
        self.global_conv = nn.Conv1d(1, 1, kernel_size=(seq_len))
        self.week_conv = nn.Conv1d(1, 1, kernel_size=(7))
        self.biweek_conv = nn.Conv1d(1, 1, kernel_size=(14))
        self.month_conv = nn.Conv1d(1, 1, kernel_size=(24)) #yea, yea, a month is not 28 days. But quadweek is a big word. 
        self.bimonth_conv = nn.Conv1d(1, 1, kernel_size=(48)) 

        self.last_week_conv = nn.Conv1d(1, 1, kernel_size=(7))
        self.last_biweek_conv = nn.Conv1d(1, 1, kernel_size=(14))
        
        self.drop_large = nn.Dropout(0.35)
        self.drop_small = nn.Dropout(0.15)
        
    def forward(self, x):
        x = x[:, None, :] #insert 1 channel (bs, channel, timesteps)
        bs = x.shape[0]
        
        out1 = self.global_conv(x).view(bs, -1)
        out1 = self.drop_large(out1)
        
        out2 = self.week_conv(x).view(bs, -1)
        out2 = self.drop_large(out2)
        
        out3 = self.biweek_conv(x).view(bs, -1)
        out3 = self.drop_large(out3)
        
        out4 = self.month_conv(x).view(bs, -1)
        out4 = self.drop_large(out4)
        
        out6 = self.last_week_conv(x[:, :, -7:]).view(bs, -1)
        out6 = self.drop_small(out6)     
        
        out7 = self.last_biweek_conv(x[:, :, -14:]).view(bs, -1)
        out7 = self.drop_small(out7)             
        
        out = torch.cat([out1, out2, out3, out4, out6, out7], axis=1)
        return out

class M5Net(nn.Module):
    def __init__(self, emb_dims, n_cont, seq_len=56, device=device):
        super().__init__()
        self.device = device
        self.convs = ConvModule(seq_len)

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        n_embs = sum([y for x, y in emb_dims])
        
        self.n_embs = n_embs
        self.n_cont = n_cont
        
        inp_dim = 384 #got this via an error, todo: write a lazy loader
#         hidden_dim = 300
        hidden_dim = inp_dim//2
        
        self.fn = nn.Sequential(
                 LinearBlock(inp_dim, hidden_dim),
                 LinearBlock(hidden_dim, hidden_dim//2),
                 LinearBlock(hidden_dim//2, hidden_dim//4),
        )          
        
        self.out = nn.Linear(hidden_dim//4, 1)

        self.fn.apply(init_weights)
        self.out.apply(init_weights)
        

    def encode_and_combine_data(self, cont_data, cat_data):
        xcat = [el(cat_data[:, k]) for k, el in enumerate(self.emb_layers)]
        xcat = torch.cat(xcat, 1)
        x = torch.cat([xcat, cont_data], axis=1)
        return x   
    
    def forward(self, cont_data, cat_data, hist_data):
        cont_data = cont_data.to(self.device)
        cat_data = cat_data.to(self.device)
        hist_data = hist_data.to(self.device)
        
        x1 = self.encode_and_combine_data(cont_data, cat_data)
        x2 = self.convs(hist_data)
        x = torch.cat([x1, x2], axis=1)
        x = self.fn(x)
        x = self.out(x)
        return x