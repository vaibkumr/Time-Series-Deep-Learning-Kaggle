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

class M5Loader:
    def __init__(self, X, y, sales_df, shuffle=True, 
                 batch_size=10000, seq_len=56, 
                 cat_cols=[], ret_garbage=False,
                 reduce=None):
        
        if reduce is not None:
            n = X["dense1"].shape[0]
            k = int((1-reduce)*n)
            reduced_idxs = np.random.choice([i for i in range(n)], k, replace=False)
            
        self.X_cont = X["dense1"]
        self.X_cat = np.concatenate([X[k] for k in cat_cols], axis=1)
        self.ids = X["id"].values.flatten()
        self.ds = X["d"].values.flatten()
        self.y = y
        
        if reduce is not None:
            self.X_cont = self.X_cont[[reduced_idxs]]
            self.X_cat = self.X_cat[[reduced_idxs]]
            self.ids = self.ids[[reduced_idxs]]
            self.ds = self.ds[[reduced_idxs]]
            self.y = self.y[[reduced_idxs]]

        self.sales_df = sales_df
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_conts = self.X_cont.shape[1]
        self.len = self.X_cont.shape[0]
        n_batches, remainder = divmod(self.len, self.batch_size)
        
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        self.remainder = remainder #for debugging
        
        self.idxes = np.array([i for i in range(self.len)])
        self.ret_garbage = ret_garbage #For last 28/56 days of test set which cant be predicted right now
        self.always_garbage = False
        
        
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            ridxes = self.idxes
            np.random.shuffle(ridxes)
            self.X_cat = self.X_cat[[ridxes]]
            self.X_cont = self.X_cont[[ridxes]]
            self.ids = self.ids[[ridxes]]
            self.ds = self.ds[[ridxes]]
            if self.y is not None:
                self.y = self.y[[ridxes]]
                
        return self

    def __next__(self):
        if self.i  >= self.len:
            raise StopIteration
        
        if self.always_garbage:
            self.i += self.batch_size
            return None, None, None, None
            
        if self.y is not None:
            y = torch.FloatTensor(self.y[self.i:self.i+self.batch_size].astype(np.float32))
        else:
            y = None
        
         
        ids = self.ids[self.i:self.i+self.batch_size]      
        ds = self.ds[self.i:self.i+self.batch_size]
        cur_batch_size = ids.shape[0]
        hist = np.zeros((cur_batch_size, self.seq_len))
        horizon = 28 
        
        if self.ret_garbage:
            try:
                for past in range(self.seq_len):
                    hist[:, past] = self.sales_df.lookup(ids, ds-horizon-self.seq_len+past) #TODO: pandas lookup is slow, maybe hash ids and do a npy lookup
            except:
                print("NOOOOOOO This should not happen")
                self.always_garbage = True
                return None, None, None, None
        else:
              for past in range(self.seq_len):
                    hist[:, past] = self.sales_df.lookup(ids, ds-horizon-self.seq_len+past) #TODO: pandas lookup is slow, maybe hash ids and do a npy lookup
                    
        xcont = torch.FloatTensor(self.X_cont[self.i:self.i+self.batch_size])
        xcat = torch.LongTensor(self.X_cat[self.i:self.i+self.batch_size])
        xhist = torch.FloatTensor(hist)
        
        batch = (xcont, xcat, xhist, y)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches           
 
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


def get_loaders(bs, shuffle, seq_len, reduce, test=False):
    # bs = 2**11
    # shuffle = True
    # seq_len = int(28*4)
    # reduce = None
    cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
    cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 
                            "event_type_1", "event_name_2", "event_type_2"]

    X_test = utils.load("X_test.tmp")
    valid = utils.load("valid.tmp")
    X_train = utils.load("X_train_final.tmp")
    y_train = utils.load("y_train_final.tmp")
    test = utils.load("test.tmp")
    fday = X_train['d'].values.flatten().min()
    path = "../data"

    sales_df = pd.read_csv(os.path.join(path, "sales_train_evaluation.csv"))
    sales_df.id = sales_df.id.apply(lambda x : "_".join(x.split("_")[:-1]))
    sales_df.index = sales_df.id
    sales_df = sales_df.drop(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1)
    cols = [i for i in range(1941)]
    sales_df.columns = cols
    sales_df.head()            

    if not test:
        train_loader = M5Loader(X_train, y_train.values, 
                                sales_df, cat_cols=cat_cols, 
                                batch_size=bs, seq_len=seq_len,
                                shuffle=shuffle, reduce=reduce)

        val_loader = M5Loader(valid[0], valid[1].values,
                                sales_df, cat_cols=cat_cols, 
                                batch_size=bs, seq_len=seq_len,
                                shuffle=False)
        print(f"Train loader len: {train_loader.len}")
        print(f"Val loader len: {val_loader.len}")
        return train_loader, val_loader                         

    X_test['id'].id = X_test['id'].id.apply(lambda x : "_".join(x.split("_")[:-1]))
    test_loader = M5Loader(X_test, y=None,
                            sales_df=sales_df, cat_cols=cat_cols, 
                            batch_size=NUM_ITEMS, seq_len=seq_len,
                            shuffle=False, ret_garbage=False)

    print(f"Test loader len: {test_loader.len}")
    return test_loader



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
