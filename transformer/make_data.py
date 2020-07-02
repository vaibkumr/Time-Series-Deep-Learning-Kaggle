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

from utils import * 


NUM_ITEMS = 30490
DAYS_PRED = 28

seed_everything() #Reproducibility

def prep_calendar(df):
    df = df.drop(["date", "weekday"], axis=1)
    df = df.assign(d = df.d.str[2:].astype(int))
    df = df.fillna("missing")
    cols = list(set(df.columns) - {"wm_yr_wk", "d"})
    df[cols] = OrdinalEncoder(dtype="int").fit_transform(df[cols])
    df = reduce_mem_usage(df)
    return df

def prep_selling_prices(df):
    gr = df.groupby(["store_id", "item_id"])["sell_price"]
    df["sell_price_rel_diff"] = gr.pct_change()
    df["sell_price_roll_sd7"] = gr.transform(lambda x: x.rolling(7).std())
    df["sell_price_roll_sd28"] = gr.transform(lambda x: x.rolling(28).std())
    df["sell_price_cumrel"] = (gr.shift(0) - gr.cummin()) / (1 + gr.cummax() - gr.cummin())
    df = reduce_mem_usage(df)
    return df

def long_sale_history_year(df):
    gr = df.groupby(['id'])['demand']
    for year in tqdm(range(1, 4)):
        for d in [-3, -2, -1, 0, 1, 2, 3]: 
            shift = year * 365 + d
            df[f'lag_year_{year}_{d}'] = gr.transform(lambda x: x.shift(shift))
    df = reduce_mem_usage(df)    
    return df

def long_sale_history_halfyear(df):
    gr = df.groupby(['id'])['demand']
    for year in tqdm(range(1, 4)):
        for d in [-3, -2, -1, 0, 1, 2, 3]: 
            shift = year * (365//2) + d
            df[f'lag_hyear_{year}_{d}'] = gr.transform(lambda x: x.shift(shift))
    df = reduce_mem_usage(df)        
    return df

def long_sale_history_quarteryear(df):
    gr = df.groupby(['id'])['demand']
    for year in tqdm(range(1, 4)):
        for d in [-3, -2, -1, 0, 1, 2, 3]: 
            shift = year * (365//4) + d
            df[f'lag_qyear_{year}_{d}'] = gr.transform(lambda x: x.shift(shift))
    df = reduce_mem_usage(df)         
    return df

def long_price_history_year(df):
    gr = df.groupby(['id'])['sell_price']
    for year in tqdm(range(1, 4)):
        for d in [-3, -2, -1, 0, 1, 2, 3]: 
            shift = year * 365 + d
            df[f'lag_year_{year}_{d}'] = gr.transform(lambda x: x.shift(shift))
    df = reduce_mem_usage(df)    
    return df


def long_price_history_halfyear(df):
    gr = df.groupby(['id'])['sell_price']
    for year in tqdm(range(1, 4)):
        for d in [-3, -2, -1, 0, 1, 2, 3]: 
            shift = year * (365//2) + d
            df[f'lag_year_{year}_{d}'] = gr.transform(lambda x: x.shift(shift))
    df = reduce_mem_usage(df)    
    return df


def long_price_history_quarteryear(df):
    gr = df.groupby(['id'])['sell_price']
    for year in tqdm(range(1, 4)):
        for d in [-3, -2, -1, 0, 1, 2, 3]: 
            shift = year * (365//4) + d
            df[f'lag_year_{year}_{d}'] = gr.transform(lambda x: x.shift(shift))
    df = reduce_mem_usage(df)    
    return df

def make_data(i, days, sales, train=False):
    if train: start = 0 
    else: start = (days + i*28)*NUM_ITEMS
    end = i*28*NUM_ITEMS
    if i == 0:
        return sales.iloc[-start:]
    return sales.iloc[-start:-end]


def save(x, fname):
    with open(fname, "wb") as handle:
        pickle.dump(x, handle)

if __name__ == "__main__":
    path = "../data"
    calendar = pd.read_csv(os.path.join(path, "calendar.csv"))
    selling_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))
    sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
    sales = pd.read_csv(os.path.join(path, "sales_train_evaluation.csv")) #For final submission
    
    calendar = prep_calendar(calendar)
    selling_prices = prep_selling_prices(selling_prices)
    dropd = 509 #Will use 4 years data only (by dropping 509 days)
    sales = reshape_sales(sales, dropd)
    #Apply functions in part to save memory
    sales = long_sale_history_year(sales)
    sales = long_sale_history_halfyear(sales)
    sales = long_sale_history_quarteryear(sales)
    # sales = long_price_history_quarteryear(sales)
    # sales = long_price_history_halfyear(sales)
    # sales = long_price_history_year(sales)
    #Merge calender
    sales = sales.merge(calendar, how="left", on="d")
    gc.collect()
    #Merge selling price
    sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
    gc.collect()
    sales.drop(["wm_yr_wk"], axis=1, inplace=True)
    cat_id_cols = ["item_id", "dept_id", "store_id", "cat_id", "state_id"]
    cat_cols = cat_id_cols + ["wday", "month", "year", "event_name_1", 
                            "event_type_1", "event_name_2", "event_type_2"]

    for i, v in tqdm(enumerate(cat_id_cols)):
        sales[v] = OrdinalEncoder(dtype="int").fit_transform(sales[[v]])

    sales = reduce_mem_usage(sales)
    gc.collect()
    num_cols_price = ["sell_price", "sell_price_rel_diff", "sell_price_roll_sd7", 
                "sell_price_roll_sd28", "sell_price_cumrel"]

    lag_cols = []
    for lag in  ['year', 'hyear', 'qyear']:
        for y in range(1, 4):
            for t in [-3, -2, 1, 0, 1, 2,  3]:
                lag_cols.append(f"lag_{lag}_{y}_{t}")

    bool_cols = ["snap_CA", "snap_TX", "snap_WI"]

    cont_cols = num_cols_price + lag_cols + bool_cols

    for i, v in enumerate(tqdm(cont_cols)):
        sales[v] = sales[v].fillna(sales[v].median())

    e_days = 28*4
    d_days = 28
    days = e_days + d_days
    test = make_data(0, days, sales)
    val = make_data(1, days, sales)
    train = make_data(2, days, sales, True)     

    print("Train data")
    print(train.shape)
    print(train.shape[0]/NUM_ITEMS)

    print("Val data")
    print(val.shape)
    print(val.shape[0]/NUM_ITEMS)

    print("Test data")
    print(test.shape)
    print(test.shape[0]/NUM_ITEMS)   

    #Sanity check
    print(">> train")
    for col in train.columns:
        if train[col].isna().any():
            print(col)
            
    print(">> val")
    for col in val.columns:
        if val[col].isna().any():
            print(col)    
            
    print(">> test")
    for col in test.columns:
        if test[col].isna().any():
            print(col)   

    save(train, "train_all.data")        
    save(val, "val.data")        
    save(test, "test.data")                     
                