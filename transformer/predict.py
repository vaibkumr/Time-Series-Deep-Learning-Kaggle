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
    model_name = "seq2seq_v_final_idgaf_anymore_59.pth"
    model = utils.load(model_name)        
    criterion = RMSE()
    
    train_loader, val_loader = get_test_loader()

    with torch.no_grad():
            model.eval()
            ytrue = []
            ypreds = []
                
            for i, (e_cont, e_cat, d_cont, d_cat, ids, _) in enumerate(tqdm(test_loader)):                
                    x = (e_cont, e_cat, d_cont, d_cat)
                    out = model(x)
                    ypreds += list(out.detach().cpu().numpy().flatten())
                    
            zc = zero_percentage(ypreds)
    print(f"zc: {zc}")     
    preds = np.array(ypreds).reshape(NUM_ITEMS, 28)
    print(pred.max())
    print(pred.min())
    print(pred.mean())
    print((pred[:NUM_ITEMS*28]<1).sum()/pred[:NUM_ITEMS*28].shape[0])
    path = "../data/"
    sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
    test["demand"] = pred.clip(0)
    submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]
    submission = sample_submission[["id"]].merge(submission, how="left", on="id")
    submission.head()    
    submission.to_csv("/home/timetraveller/Desktop/submit_this.csv", index=False)

    