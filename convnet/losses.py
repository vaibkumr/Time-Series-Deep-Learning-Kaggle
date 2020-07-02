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
from tqdm import tqdm_notebook as tqdm
import utils


calc_wrmsse = True

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        y_true = y_true.type(torch.FloatTensor).to(device).flatten()
        y_pred = y_pred.flatten()
        return self.mse(y_pred, y_true)

class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        y_true = y_true.type(torch.FloatTensor).to(device).flatten()
        y_pred = y_pred.flatten()
        return torch.sqrt(self.mse(y_pred, y_true))    
    
class Assymetric_RMSE(nn.Module):
    def __init__(self, penalty=3.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.penalty = penalty
        
    def forward(self, y_pred, y_true):
        y_true = y_true.type(torch.FloatTensor).to(device).flatten()
        y_pred = y_pred.flatten()
        error = torch.where(y_true==0, self.penalty*(y_true-y_pred)**2, (y_true-y_pred)**2)
        return torch.sqrt(torch.mean(error))

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        y_true = y_true.type(torch.FloatTensor).to(device).flatten()
        y_pred = y_pred.flatten()
        return torch.mean(torch.abs(y_pred-y_true))        
    
if calc_wrmsse:
    roll_mat_df = pd.read_pickle('../data/roll_mat_df.pkl')
    roll_index = roll_mat_df.index
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df; gc.collect()

    sw_df = pd.read_pickle('../data/sw_df.pkl')
    s = sw_df.s.values
    w = sw_df.w.values
    sw = sw_df.sw.values   
    
def rollup(v, roll_mat_csr):
    return roll_mat_csr*v #(v.T*roll_mat_csr.T).T

def wrmsse_metric(preds, y_true, score_only=True, npy=True, roll_mat_csr=None, sw=None, roll_index=roll_index, verbose=False, s=s, w=w):
    preds = np.array(preds).reshape(NUM_ITEMS, -1)
    y_true = np.array(y_true).reshape(NUM_ITEMS, -1)

    if roll_mat_csr is None:
        roll_mat_df = pd.read_pickle('../data/roll_mat_df.pkl')
        roll_index = roll_mat_df.index
        roll_mat_csr = csr_matrix(roll_mat_df.values)
        del roll_mat_df; gc.collect()

    if sw is None:
        sw_df = pd.read_pickle('../data/sw_df.pkl')
        s = sw_df.s.values
        w = sw_df.w.values
        sw = sw_df.sw.values

    if not npy:
        preds = preds.values
        y_true = y_true.values
    
    if score_only:
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(rollup(preds-y_true, roll_mat_csr))
                            ,axis=1)) * sw)/12 
    else: 
        score_matrix = (np.square(rollup(preds-y_true, roll_mat_csr)) * np.square(w)[:, None])/ s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix, axis=1)))/12 
        
        score_df = pd.DataFrame(score_matrix, index = roll_index)
        score_df.reset_index(inplace=True)
        score_df['mean_err'] = score_df.mean(axis=1)
        level_wise_error1 = score_df.groupby('level')['mean_err'].mean()
        
        return score, level_wise_error1.values

def rmse_metric(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.sqrt(np.mean((y_pred-y_true)**2))                                

def rmse_metric_npy(y_pred, y_true):
    return np.sqrt(np.mean((y_pred-y_true)**2))  