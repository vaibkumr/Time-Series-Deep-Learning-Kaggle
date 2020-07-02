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

import utils


NUM_ITEMS = 30490
DAYS_PRED = 28

utils.seed_everything() #Reproducibility

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

def prep_sales(df):
    df['lag_t28'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    df['lag_t90'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(90))
    df['lag_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(180))
    df['lag_t365'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(365))
    df['rolling_mean_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['rolling_mean_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    df['rolling_mean_t60'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
    df['rolling_mean_t90'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    df['rolling_mean_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    df['rolling_std_t7'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    df['rolling_std_t30'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    df['rolling_std_t90'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).std())

    # Remove rows with NAs except for submission rows. rolling_mean_t180 was selected as it produces most missings
    df = df[(df.d >= 1941) | (pd.notna(df.rolling_mean_t180))]
    df = reduce_mem_usage(df)

    return df

def make_X(df):
    X = {"dense1": df[dense_cols].to_numpy()}
    for i, v in enumerate(cat_cols):
        X[v] = df[[v]].to_numpy()
    X['id'] = df[['id']]        
    X['d'] = df[['d']]        
    return X

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
    sales = prep_sales(sales)
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
    num_cols = ["sell_price", "sell_price_rel_diff", "sell_price_roll_sd7", "sell_price_cumrel",
            "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", 
            "rolling_mean_t90", "rolling_mean_t180", "rolling_std_t7", "rolling_std_t30"]
    bool_cols = ["snap_CA", "snap_TX", "snap_WI"]

    dense_cols = num_cols + bool_cols

    # Need to do column by column due to memory constraints
    for i, v in tqdm(enumerate(num_cols)):
        sales[v] = sales[v].fillna(sales[v].median())

    test = sales[sales.d >= 1914]
    test = test.assign(id=test.id + "_" + np.where(test.d <= 1941, "validation", "evaluation"),
                    F="F" + (test.d - 1913 - 28 * (test.d > 1941)).astype("str"))
    test.head()
    gc.collect()     

    X_test = make_X(test)

    # One month of validation data
    flag = (sales.d < 1942) & (sales.d >= 1942 - 28)
    valid = (make_X(sales[flag]),
            sales["demand"][flag])

    # Rest is used for training
    # flag = sales.d < 1942 - 28
    flag = sales.d < 1942 
    X_train = make_X(sales[flag])
    y_train = sales["demand"][flag]
                                
    del sales, flag
    gc.collect()    

    utils.save(X_train, "X_train_final.tmp")
    del X_train; gc.collect()
    utils.save(y_train, "y_train_final.tmp")
    del y_train; gc.collect()
    utils.save(X_test, "X_test.tmp")
    del X_test; gc.collect()
    utils.save(valid, "valid.tmp")
    del valid; gc.collect()
    utils.save(test, "test.tmp")    