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
from tqdm import tqdm
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
import utils

NUM_ITEMS = 30490
DAYS_PRED = 28


utils.seed_everything()    


class M5Dataset_train(Dataset):
    def __init__(self, X, cont_cols, cat_cols, id_cols, lag_cols, 
                             target='demand', e_days=28*4, d_days=28):
        
        self.e_days = e_days
        self.d_days = d_days
        cat_cols = id_cols + cat_cols 
        self.cat_cols = cat_cols
        
        self.X_cont = X[cont_cols].values
        self.X_cat = X[cat_cols].values
        self.X_lags = X[lag_cols].values
        self.ids = X[id_cols].values
        self.y = X[target].values
        
        x_days = X.shape[0]//NUM_ITEMS
        total_days = x_days - e_days - d_days
        self.len = int(NUM_ITEMS * total_days)
        self.total_days = total_days
        print(f"total_days: {total_days}")
        print(f"len: {self.len}")

    def get_ti(self, idx):
        time, item = divmod(idx, NUM_ITEMS)
        return time, item
    
    def __getitem__(self, idx):
        time, item = self.get_ti(idx)
        e_start_day = time
        e_end_day = time + self.e_days
        d_start_day = time + self.e_days
        d_end_day = time + self.e_days + self.d_days
        
        e_idxes = item + np.arange(e_start_day, e_end_day)*NUM_ITEMS
        d_idxes = item + np.arange(d_start_day, d_end_day)*NUM_ITEMS

        #Encoding cat and cont information
        encoder_cont = np.concatenate([self.X_cont[e_idxes],
                                        self.y[e_idxes].reshape(-1, 1)], 
                                        axis=1)
        encoder_cat = self.X_cat[e_idxes]
        encoder_lags = self.X_lags[e_idxes]
        enc_hist = self.y[e_idxes]
        
        #Decoding cat and cont information
        decoder_cont = self.X_cont[d_idxes]
        decoder_cat = self.X_cat[d_idxes]
        decoder_lags = self.X_lags[d_idxes]
        
        #Labels
        labels = self.y[d_idxes]
        #Id (same for all timesteps)
        ids = self.ids[idx]
        return (enc_hist, encoder_cont, encoder_cat, encoder_lags, 
                        decoder_cont, decoder_cat, decoder_lags), idx, labels

    def __len__(self):
        return self.len

    def __repr__(self):
        return "e_cont, e_cat, d_cont, d_cat, ids, y"

class M5Dataset(Dataset):
    def __init__(self, X, cont_cols, cat_cols, id_cols, lag_cols, 
                             target='demand', e_days=28*4, d_days=28):
        
        self.e_days = e_days
        self.d_days = d_days
        cat_cols = id_cols + cat_cols 
        self.cat_cols = cat_cols
        
        self.X_cont = X.iloc[-NUM_ITEMS*(e_days+d_days):][cont_cols].values
        self.X_cat = X.iloc[-NUM_ITEMS*(e_days+d_days):][cat_cols].values
        self.X_lags = X.iloc[-NUM_ITEMS*(e_days+d_days):][lag_cols].values
        self.ids = X.iloc[-NUM_ITEMS*(e_days+d_days):][id_cols].values
        self.y = X.iloc[-NUM_ITEMS*(e_days+d_days):][target].values
        
        self.len = NUM_ITEMS

    def __getitem__(self, idx):
        item = idx
        e_start_day = 0
        e_end_day = self.e_days
        d_start_day = self.e_days
        d_end_day = self.e_days  + self.d_days      
        
        
        e_idxes = item + np.arange(e_start_day, e_end_day)*NUM_ITEMS
        d_idxes = item + np.arange(d_start_day, d_end_day)*NUM_ITEMS

        #Encoding cat and cont information
        encoder_cont = np.concatenate([self.X_cont[e_idxes],
                                        self.y[e_idxes].reshape(-1, 1)], 
                                        axis=1)
        encoder_cat = self.X_cat[e_idxes]
        encoder_lags = self.X_lags[e_idxes]
        enc_hist = self.y[e_idxes]
        
        #Decoding cat and cont information
        decoder_cont = self.X_cont[d_idxes]
        decoder_cat = self.X_cat[d_idxes]
        decoder_lags = self.X_lags[d_idxes]
        
        #Labels
        labels = self.y[d_idxes]
        #Id (same for all timesteps)
        ids = self.ids[idx]
        return (enc_hist, encoder_cont, encoder_cat, encoder_lags, 
                        decoder_cont, decoder_cat, decoder_lags), idx, labels

    def __len__(self):
        return self.len

    def __repr__(self):
        return "e_cont, e_cat, d_cont, d_cat, ids, y"        

def load(fname):
    with open(fname, "rb") as handle:
        return pickle.load(handle)
    
 
def get_cols():
    id_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    cat_cols = ['wday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    num_cols_price = ["sell_price", "sell_price_rel_diff", "sell_price_roll_sd7", "sell_price_roll_sd28", "sell_price_cumrel"]

    lag_cols = []
    for lag in  ['year', 'hyear', 'qyear']:
        for y in range(1, 4):
            for t in [-3, -2, 1, 0, 1, 2,  3]:
                lag_cols.append(f"lag_{lag}_{y}_{t}")

    bool_cols = ["snap_CA", "snap_TX", "snap_WI"]

    cont_cols = num_cols_price + bool_cols

    print(f"# Cat cols: {len(cat_cols)}")
    print(f"# Id cols: {len(id_cols)}")
    print(f"# Cont cols: {len(cont_cols)}")
    print(f"# Lag cols: {len(lag_cols)}")
    return cont_cols, cat_cols, id_cols, lag_cols


def get_loaders(bs, shuffle, num_workers=4, e_days=28*4, d_days=28):
    # bs = 128
    # shuffle = True
    # num_workers = 4
    # e_days = 28*4 #Can't be more than 28*4 for now
    # d_days = 28

    # train = load("train.data")   #Validation
    train = load("train_all.data") #Final submit
    val = load("val.data")    
    test = load("test.data")   
    cont_cols, cat_cols, id_cols, lag_cols = get_cols()

    train_dataset = M5Dataset_train(train, cont_cols, cat_cols, 
                                id_cols, lag_cols, e_days=e_days, 
                                d_days=d_days)

    train_loader = DataLoader(train_dataset, batch_size=bs,
                                shuffle=shuffle, num_workers=num_workers)

    val_dataset = M5Dataset(val, cont_cols, cat_cols, 
                                id_cols, lag_cols, 
                                e_days=e_days, d_days=d_days)

    val_loader = DataLoader(val_dataset, batch_size=bs,
                                shuffle=False, num_workers=num_workers)


def get_test_loader():
    X_test['id'].id = X_test['id'].id.apply(lambda x : "_".join(x.split("_")[:-1]))
    test_loader = M5Loader(X_test, y=None,
                sales_df=sales_df, cat_cols=cat_cols, 
                batch_size=NUM_ITEMS, seq_len=seq_len,
                shuffle=False, ret_garbage=True)


# if __name__ == "__main__":
#     """unit tests"""
#     print("Train")
#     print(len(train_loader))

#     for i, (x, idx, y) in enumerate(train_loader):
#         for _x in x:
#             print(_x.shape)
#         print(idx)    
#         break
        
#     print("\nVal")
#     print(len(val_loader))

#     for i, (x, idx, y) in enumerate(val_loader):
#         for _x in x:
#             print(_x.shape)
#         print(idx)    
#         break    
