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
from torch.utils.data import TensorDataset, DataLoader, Dataset

from losses import *
from model import *
from loader import *
import utils

utils.seed_everything()

NUM_ITEMS = 30490
            
def zero_percentage(q):
    l = len(q)
    q = np.array(q)
    return sum(q<1)/l       

if __name__ == "__main__":
    d_input = 44 
    d_model = 64 
    d_output = 1
    q = 8  
    v = 8  
    h = 2 # Number of heads
    N_e = 3 # Number of encoders to stack
    N_d = 1 # Number of decoders to stack
    attention_size = 12  
    n_months = e_days // d_days
    dropout = 0.2 
    pe = None
    chunk_mode = None
    n_lag_convs = 9 #over 63 past lags
    torch.manual_seed(777)
    model = M5Transformer(d_input, d_model, d_output, q, v, h, 
            N_e, N_d, n_months, attention_size=attention_size,
            dropout=dropout, chunk_mode=chunk_mode, pe=pe, 
            n_lag_convs=n_lag_convs, e_days=e_days,
            d_days=d_days).to(device)
    epochs = 60
    lr = 3e-4
    # criterion = Assymetric_RMSE(penalty=1/3.5)
    # criterion = Assymetric_RMSE(penalty=3.5)
    # criterion = WRMSSE() #WARNING! RAM EXPLOSION HERE
    # criterion = RMSE_summed()
    criterion = RMSE()
    # criterion = WeightedRoll_MSE() #WARNING! RAM EXPLOSION HERE
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 20, T_mult=2, 
                    eta_min=lr/1e3, last_epoch=-1)

    model_name = "seq2seq_v_final_idgaf_anymore.pth"
    
    train_loader, val_loader = get_loaders(bs, shuffle, 
                num_workers=4, e_days=28*4, d_days=28)

    # Start training
    train_zc, val_zc = None, None
    train_losses = []
    val_losses = []
    train_wrmsses = []
    val_wrmsses = []

    for epoch in tqdm(range(epochs)):
        train_loss, val_loss = 0, 0
        train_rmse = 0
        #Training phase
        model.train()
        ypreds = [] 
        ytrue = []
        bar = tqdm(train_loader)
        for i, (x, idx, y) in enumerate(bar):
            optimizer.zero_grad()
            out = model(x)
    #         loss = criterion(out, y, idx)   #This is for WRMSSE related losses
            loss = criterion(out, y)   
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i/len(train_loader))
            
            with torch.no_grad():
                train_loss += loss.item()/len(train_loader)
                ypreds = out.detach().cpu().numpy().flatten()
                ytrue = y.cpu().numpy().flatten()
                train_rmse += rmse_metric_npy(ypreds, ytrue)/len(train_loader)
                bar.set_description(f"{loss.item():.3f}")
        
        train_wrmsse = 0.5 #whatever
        print(f"[Train] Epoch: {epoch} | Loss: {train_loss:.4f} | RMSE: {train_rmse:.4f}")
        
        #Validation phase      
        with torch.no_grad():
                    model.eval() #smetimes .eval gives worse results even with dropout and batchnorm (I have no idea why!)
                    ytrue = []
                    ypreds = []

                    for i, (x, idx, y) in enumerate(val_loader):                
                        out = model(x)
    #                     loss = criterion(out, y, idx)   #This is for WRMSSE related losses
                        loss = criterion(out, y)   
                        val_loss += loss.item()/len(val_loader)
                        ypreds += list(out.detach().cpu().numpy().flatten())
                        ytrue += list(y.cpu().numpy().flatten())

                    rrmse = rmse_metric(ypreds, ytrue)
                    val_wrmsse, lws = wrmsse_metric(ypreds, ytrue, roll_mat_csr=roll_mat_csr, sw=sw, score_only=False)

                    if val_zc is None:
                        val_zc = zero_percentage(ytrue) 
                    zc = zero_percentage(ypreds)

                    print(f"[Valid] Epoch: {epoch} | Loss: {val_loss:.4f} | RMSE: {rrmse:.4f} | zc: ({zc:.3f}/{val_zc:.3f}) | wrmsse: {val_wrmsse:.4f}")
                    print(lws)

        train_losses.append(train_loss)    
        val_losses.append(val_loss)     
        train_wrmsses.append(train_wrmsse)
        val_wrmsses.append(val_wrmsse)
        utils.save(model, f"{model_name}_{epoch}")

        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.show()

        plt.plot(train_wrmsses)
        plt.plot(val_wrmsses)
        plt.show()
     