import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from utils import *

warnings.filterwarnings("ignore")

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

# prepare training and test data.
# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)


seed_everything() #Make everything deterministic 
NUM_ITEMS = 30490
DAYS_PRED = 28
pd.set_option('use_inf_as_na', True)

def read_data(dir='data'):
    print("Reading files...")
    calendar = pd.read_csv(os.path.join(dir, 'calendar.csv')).pipe(reduce_mem_usage)
    sell_prices = pd.read_csv(os.path.join(dir, 'sell_prices.csv')).pipe(reduce_mem_usage)
    sales_train_val = pd.read_csv(os.path.join(dir, 'sales_train_validation.csv')).pipe(reduce_mem_usage)
    submission = pd.read_csv(os.path.join(dir, 'sample_submission.csv')).pipe(reduce_mem_usage)
    print("calendar shape:", calendar.shape)
    print("sell_prices shape:", sell_prices.shape)
    print("sales_train_val shape:", sales_train_val.shape)
    print("submission shape:", submission.shape)

    # calendar shape: (1969, 14)
    # sell_prices shape: (6841121, 4)
    # sales_train_val shape: (30490, 1919)
    # submission shape: (60980, 29)

    return calendar, sell_prices, sales_train_val, submission

def handle_categorical(df, cols):
    for col in cols:
        # if df[col].isnull().any():
        #     df[f'{col}_isna'] = df[col].isnull() # is_na features
        #     df[f'{col}_isna'] = df[f'{col}_isna'].astype('category')
        #     df[f'{col}_isna'] = df[f'{col}_isna'].cat.codes

        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes.astype("int16")

    return df

def melt(sales_train_val, submission, nrows=1000000, verbose=True,):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # get product table.
    product = sales_train_val[id_columns]

    sales_train_val = sales_train_val.melt(
        id_vars=id_columns, var_name="d", value_name="demand",
    ).pipe(reduce_mem_usage)

    print("melted")

    # separate test dataframes.
    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    # change column names.
    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

    # merge with product table
    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(product, how="left", on="id")
    evals = evals.merge(product, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")


    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")
    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    sales_train_val["part"] = "train"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    data = pd.concat([sales_train_val, vals, evals], axis=0)

    del sales_train_val, vals, evals
    gc.collect()

    # delete evaluation for now.
    data = data[data["part"] != "evaluation"]
    gc.collect()

    data = data.iloc[-nrows:]
    return data

def extract_d(df):
    return df["d"].str.extract(r"d_(\d+)").astype(np.int16)

def merge_calendar(data, calendar):
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
    gc.collect()
    return data.merge(calendar, how="left", on="d").assign(d=extract_d)

def merge_sell_prices(data, sell_prices):
    return data.merge(sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

def merge_scale(data, scale):
    """scale is the thing used on denom of calculating RMSSE"""
    return data.merge(scale, how="left", on=["id", "d"])

def merge_target_encoding(data, encodings):
    """target encoding. See `notebooks/10. Encoding.ipynb`"""
    return data.merge(encodings, how="left", on=["id", "d"])

def add_demand_features(df):
    DAYS_PRED = 28

    group = df[["demand", "id"]].groupby(["id"])

    for size in [7, 14, 28, 90]:
        print("Adding rolling feature of size: {size}")
        df[f"rolling_mean_t{size}_demand"] = group["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).mean()
        )
        df[f"rolling_std_t{size}_demand"] = group["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).std()
        )            
    
    for lag in [28, 90, 365]:
        print("Adding lagging feature of size: {lag}")
        df[f"lag_{size}_demand"] = group["demand"].transform(
            lambda x: x.shift(lag)
        )
    
    df["demand_max"] = group["demand"].transform(
        lambda x: x.shift(DAYS_PRED).max()
    )
    
    df["demand_min"] = group["demand"].transform(
        lambda x: x.shift(DAYS_PRED).min()
    )

    df["demand_nunique"] = group["demand"].transform(
        lambda x: x.nunique()
    ) 

    # df["demand_dist"] = (df["demand_max"] - df["demand_min"]) / df["demand_nunique"]

    print("Demand features added")
    return df

def add_demand_features_minimal(df, recursive=False):

    if recursive:
        df["non_zeros"] = (df["demand"].round()>0).astype(np.int8) #lightgbm wont ever predict exact zeros so round it.
    else:
        df["non_zeros"] = (df["demand"]>0).astype(np.int8)

    df["lagged_non_zeros"] = df.groupby(["id"])["non_zeros"].transform(lambda x: x.shift(1).rolling(2000, 1).sum()).fillna(-1) 
    temp_df = df[['id','d','lagged_non_zeros']].drop_duplicates(subset=['id','lagged_non_zeros'])
    temp_df.columns = ['id','d_min','lagged_non_zeros']
    df = df.merge(temp_df, on=['id','lagged_non_zeros'], how='left')
    
    del temp_df
    gc.collect()

    df['last_sale'] = df['d'] - df['d_min']
    df.drop(["non_zeros", "lagged_non_zeros", "d_min"], axis=1, inplace=True)
    gc.collect()  

    print("last_sale added!")
    
    lags = [7, 14, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    group = df[["id","demand"]].groupby("id")["demand"]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = group.shift(lag)

    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            group = df[["id", lag_col]].groupby("id")[lag_col]
            df[f"rmean_{lag}_{win}"] = group.transform(lambda x : x.rolling(win).mean())
            # df[f"rstd_{lag}_{win}"] = group.transform(lambda x : x.rolling(win).std())
    
    print("Minimal demand features added")
    return df

def target_encoding(df):
    TARGET = "demand"
    icols =  [
                ['state_id'],
                ['store_id'],
                ['cat_id'],
                ['dept_id'],
                ['item_id'],
                # ["event_name_1"],
                # ["event_name_2"],
                # ["event_type_1"],
                # ["event_type_2"],
                # ['state_id', 'cat_id'],
                # ['state_id', 'dept_id'],
                # ['store_id', 'cat_id'],
                # ['store_id', 'dept_id'],
                # ['item_id', 'state_id'],
                # ['item_id', 'store_id']
            ]

    for col in icols:
        print('Encoding', col)
        col_name = '_'+'_'.join(col)+'_'
        group = df.groupby(col)[TARGET]
        df['enc'+col_name+'mean'] = group.transform('mean').astype(np.float16)
        # df['enc'+col_name+'std'] = group.transform('std').astype(np.float16)

    drop_cols = ["store_id", "cat_id", "state_id", "dept_id", "item_id"]
    print("Target encoding done")
    # return df.drop(drop_cols, axis=1)    
    return df    

def add_price_features(df):
    group = df[["id", "sell_price"]].groupby(["id"])
    # df["price_max"] = group["sell_price"].transform('max')
    # df["price_min"] = group["sell_price"].transform('min')
    # df["price_mean"] = group["sell_price"].transform('mean')
    # df["price_std"] = group["sell_price"].transform('std')
    df["price_unique"] = group["sell_price"].transform('nunique')

    # df["price_norm2"] = df["sell_price"]/df["price_max"]
    
    df['price_momentum'] = df['sell_price']/group['sell_price'].transform(lambda x: x.shift(1))
    
    df["shift_price_t1"] = group["sell_price"].transform(
        lambda x: x.shift(1))

    df["price_change_t1"] = (df["shift_price_t1"] - df["sell_price"]) / (
        df["shift_price_t1"])

    df["shift_price_t7"] = group["sell_price"].transform(
        lambda x: x.shift(7))

    df["price_change_t7"] = (df["shift_price_t7"] - df["sell_price"]) / (
        df["shift_price_t7"])
        
    print("Price features added")
    # return df
    return df.drop(["shift_price_t1", "shift_price_t7"], axis=1)

def add_price_demand_features(df):
    DAYS_PRED = 28

    df["d_by_p"] = df["demand"]/df["sell_price"]
    #shifts to prevent leak
    group = df.groupby(["id"])["d_by_p"]
    df["d_by_p_mean"] = group.transform(
        lambda x: x.shift(DAYS_PRED).mean()
    )
    df["d_by_p_std"] = group.transform(
        lambda x: x.shift(DAYS_PRED).std()
    )        
    print("price-demand features added")
    return df.drop(["d_by_p"], axis=1)

def add_time_features(df, dt_col="date"):
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "month",
        "week",
        "day",
        "dayofweek",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    df["year"] = df["year"]-df["year"].min() #I dont think this makes any difference for tree based models but... YOLO, I keep doing redundant things in life anyway.
    print("Time features added")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str,
                    help="output directory", default="/home/timetraveller/Work/m5data")
    parser.add_argument("-f", "--fname", type=str,
                    help="output filename", default="features.hdf")
    parser.add_argument("-n", "--nyears", type=float,
                    help="number of years of data to consider", default=2)
    
    args = parser.parse_args()
    out_dir = args.dir
    out_fname = args.fname
    n = args.nyears


    calendar, sell_prices, sales_train_val, submission = read_data()
    # scale = pd.read_hdf('data/denoms.hdf')
    # scale = pd.read_hdf('data/my_weights/wts1.hdf')

    calendar = handle_categorical(calendar,
                    ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
                    ).pipe(reduce_mem_usage)

    sales_train_val = handle_categorical(sales_train_val,
                    ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
                    ).pipe(reduce_mem_usage)

    sell_prices = handle_categorical(sell_prices,
                    ["item_id", "store_id"]
                    ).pipe(reduce_mem_usage)

    nrows = int(n*NUM_ITEMS*365)
    data = melt(sales_train_val, submission, nrows=nrows)
    print(f"Data: {data.shape}")

    del sales_train_val
    gc.collect()

    # data = merge_scale(data, scale)
    # del scale
    # gc.collect()

    # encodings = pd.read_csv("/home/timetraveller/Work/m5data/encoding1.csv").pipe(reduce_mem_usage)
    # encodings = pd.read_hdf("/home/timetraveller/Work/m5data/encoding.hdf").pipe(reduce_mem_usage)
    # print(f"encodings: {encodings.shape}")
    # data = merge_target_encoding(data, encodings)
    # print(f"Data: {data.shape}")
    # del encodings
    # gc.collect()
    
    data = merge_calendar(data, calendar)
    del calendar
    gc.collect()

    data = merge_sell_prices(data, sell_prices)
    del sell_prices
    gc.collect()

    # data[f'sell_price_isna'] = data['sell_price'].isnull()
    # data[f'sell_price_isna'] = data['sell_price_isna'].astype('category')
    # data[f'sell_price_isna'] = data['sell_price_isna'].cat.codes
    
    dt_col = "date"
    
    # data = add_demand_features_minimal(data).pipe(reduce_mem_usage)
    # gc.collect()

    # data = target_encoding(data).pipe(reduce_mem_usage) #There's (might be, most probably) a leakage here.
    # gc.collect()
    
    data = add_demand_features(data).pipe(reduce_mem_usage)
    gc.collect()
    
    data = add_price_features(data).pipe(reduce_mem_usage)
    gc.collect()

    data = add_price_demand_features(data).pipe(reduce_mem_usage)
    gc.collect()

    data = add_time_features(data).pipe(reduce_mem_usage)
    gc.collect()
    
    data = data.sort_values("date")
    gc.collect()

    #Drop NA rows
    #data.dropna(inplace = True)

    print("start date:", data[dt_col].min())
    print("end date:", data[dt_col].max())
    print("data shape:", data.shape)

    # data.to_csv(os.path.join(our_dir, out_fname), index=False)
    data.to_hdf(os.path.join(out_dir, out_fname), key='df', mode='w')
    print(f"====== File saved at: {os.path.join(out_dir, out_fname)} ======")
    del data
    gc.collect()