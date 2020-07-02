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
        model_name = "final_submit_39"
        model = utils.load(model_name)        
        criterion = RMSE()
        
        bs = NUM_ITEMS 
        shuffle = False
        seq_len = int(28*4) #Hist data len in no. of days
        reduce = None
        test_loader = get_loaders(bs, shuffle, seq_len, reduce, test=True)
        pred = []
        with torch.no_grad():
                model.eval()
                for i, (X_cont, X_cat, X_hist, y) in enumerate(tqdm(test_loader)):
                if X_cont is None:
                        out_npy = np.zeros_like(out_npy)
                else:    
                        out = model(X_cont, X_cat, X_hist)
                        out_npy = out.cpu().numpy().flatten()
                pred += list(out_npy)    
        
        
        pred = np.array(pred)    
        print(pred.max())
        print(pred.min())
        print(pred.mean())
        print((pred[:NUM_ITEMS*28]<1).sum()/pred[:NUM_ITEMS*28].shape[0])
        print((pred<1).sum()/pred.shape[0])        
        path = "../data/"
        sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
        test["demand"] = pred.clip(0)
        submission = test.pivot(index="id", columns="F", values="demand").reset_index()[sample_submission.columns]
        submission = sample_submission[["id"]].merge(submission, how="left", on="id")
        submission.to_csv("/home/timetraveller/Desktop/submit_this_final.csv", index=False)