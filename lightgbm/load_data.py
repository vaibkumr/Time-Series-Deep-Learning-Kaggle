from utils import *
import gc
import os
import warnings
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import argparse

warnings.filterwarnings("ignore")
seed_everything()
NUM_ITEMS = 30490
DAYS_PRED = 28
YEARS = 1

class TimeSeriesCV:
    def __init__(self, n_splits=5, predict_days=28, date_col="date"):
        self.n_splits = n_splits
        self.predict_days = predict_days
        self.date_col = date_col

    def split(self, X, y=None, groups=None):
        X = X[self.date_col]
        # print(f"CV: {X.shape}")
        for i in range(self.n_splits):
            offset_train = (self.n_splits-i)*self.predict_days
            offset_val = offset_train - self.predict_days

            end_of_time = X.iloc[-1]
            train_end = valid_start = end_of_time - pd.to_timedelta(offset_train, unit='d')
            valid_end = end_of_time - pd.to_timedelta(offset_val, unit='d')

            train_idx = X[X<train_end].index.values
            val_idx = X[(X>=valid_start) & (X<valid_end)].index.values

            # print(f"offset_train: {offset_train}")
            # print(f"offset_val: {offset_val}")
            # print(f"end_of_time: {end_of_time}")
            # print(f"train_start: {X.iloc[0]}")
            # print(f"train_end: {train_end}")
            # print(f"valid_start: {valid_start}")
            # print(f"end_of_time: {end_of_time}")

            yield train_idx, val_idx

    def __len__(self):
        return self.n_splits
