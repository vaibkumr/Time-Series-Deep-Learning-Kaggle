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




def zero_percentage(q):
    l = len(q)
    q = np.array(q)
    return sum(q<1)/l       

if __name__ == "__main__":
    bs = 2**11 #Huge batch size is nice here
    # bs = 2**7
    shuffle = True
    seq_len = int(28*4) #Hist data len in no. of days
    reduce = None
    train_loader, val_loader = get_loaders(bs, shuffle, 
                seq_len, reduce, test=False)


    #Hard coded embedding dimensions for now
    uniques = [3049, 7, 10, 3, 3, 7, 12, 6, 31, 5, 5, 5]
    dims = [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    emb_dims = [(x, y) for x, y in zip(uniques, dims)]
    n_cont = train_loader.n_conts    
    model = M5Net(emb_dims, n_cont).to(device)

    epochs = 40
    optim = "adam"
    lr_adam = 3e-4
    lr_sgd = 1e-3
    criterion = RMSE()
    torch.manual_seed(777)
    model = M5Net(emb_dims, n_cont).to(device)
    model_name = "final_submit"

    if optim == "adam":
        print("Using adam optimizer.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    else:
        print("Not using adam optimizer.")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.9)
        
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #                     optimizer, [20, 25, 30], gamma=0.5, 
    #                     last_epoch=-1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, 5, T_mult=2, 
                        eta_min=3e-5, last_epoch=-1)

    
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        train_loss, val_loss = 0, 0
    
        #Training phase
        model.train()
        ypreds = [] 
        ytrue = []
        bar = tqdm(train_loader)
        
        for i, (X_cont, X_cat, X_hist, y) in enumerate(bar):
            optimizer.zero_grad()
            out = model(X_cont, X_cat, X_hist)
            loss = criterion(out, y)   
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i/len(train_loader))

            with torch.no_grad():
                train_loss += loss.item()/len(train_loader)
                ypreds += list(out.detach().cpu().numpy().flatten())
                ytrue += list(y.cpu().numpy())
                bar.set_description(f"{loss.item():.3f}")
        
        with torch.no_grad():
            rrmse = rmse_metric(ypreds, ytrue)
            print(f"[Train] Epoch: {epoch} | Loss: {train_loss:.4f} | RMSE: {rrmse:.4f}")
        
        #Validation phase      
        with torch.no_grad():
                model.eval()
                ytrue = []
                ypreds = []
                
                for i, (X_cont, X_cat, X_hist, y) in enumerate(val_loader):
                    out = model(X_cont, X_cat, X_hist)
                    val_loss += criterion(out, y).item()/len(val_loader)
                    ypreds += list(out.detach().cpu().numpy().flatten())
                    ytrue += list(y.cpu().numpy())
                    
                rrmse = rmse_metric(ypreds, ytrue)    
                wrmsse = wrmsse_metric(ypreds, ytrue, roll_mat_csr=roll_mat_csr, sw=sw)
                zc = zero_percentage(ypreds)
                
                print(f"[Valid] Epoch: {epoch} | Loss: {val_loss:.4f} | RMSE: {rrmse:.4f} | WRMSSE: {wrmsse:.4f} | zc: {zc:.3f}/0.544")
                
                #Test's zc
                for i, (X_cont, X_cat, X_hist, y) in enumerate(test_loader):
                    out = model(X_cont, X_cat, X_hist)
                    ypreds += list(out.detach().cpu().numpy().flatten())
                zc = zero_percentage(ypreds[-NUM_ITEMS*28:]) #Last 28 days only
                print(f"[Test] Epoch: {epoch} | zc: {zc:.3f}/???")            
        
        train_losses.append(train_loss)    
        val_losses.append(val_loss)   
        save_data = {
                'model' : model,
                'optimizer' : optimizer,
                'scheduler' : scheduler,
                'epoch' : epoch,
                'train_losses' : train_losses,
                'val_losses' : val_losses,
            }
        utils.save(save_data, f"{model_name}_{epoch}")     
    
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()   