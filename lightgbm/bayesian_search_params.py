import gc
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import register_matplotlib_converters
from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from utils import *
from load_data import *
from train import *

warnings.filterwarnings("ignore")
seed_everything()
NUM_ITEMS = 30490
DAYS_PRED = 28

# path = "/home/timetraveller/Work/m5data/tiny.hdf"
path = "/home/timetraveller/Work/m5data/large_1.hdf"
data = pd.read_hdf(path).pipe(reduce_mem_usage)
print(data.shape)

# print(data.columns)
# for c in data.columns:
#     print(c, "\t", data[c].dtype)

mask = data["date"] <= "2016-04-24"

cat_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price_isna",
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
        "is_weekend",
    ]

cat_cols = [] #MEMORY PROBLEMS!!

for c in cat_cols:
    data[c] = data[c].astype('category')   


X_train = data[mask].reset_index(drop=True)
y_train = data[mask]["demand"].reset_index(drop=True)
X_test = data[~mask].reset_index(drop=True)
id_date = data[~mask][["id", "date"]].reset_index(drop=True)

del data
del mask
gc.collect()

cv_params = {'n_splits':1, 'predict_days':28,}

cv = TimeSeriesCV(**cv_params)
exclude_features = [
            'id',
            'date',
            'demand',
            'part',
            'd',
            'scale',
        ] 

def bayes_parameter_opt_lgb(X, y, cv,  init_round=25, opt_round=50):

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, bagging_freq, 
                    learning_rate, min_data_in_leaf, min_hessian, reg_alpha,
                             reg_lambda, early_stopping_rounds):

        params = {
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_jobs': 4,
                'verbosity':-1,
                'objective': None,
                'num_leaves': int(round(num_leaves)),
                'feature_fraction' : max(min(feature_fraction, 1), 0),
                'bagging_fraction': max(min(bagging_fraction, 1), 0),
                'bagging_freq': int(round(bagging_freq)),
                'learning_rate': learning_rate,
                'min_data_in_leaf': int(round(min_data_in_leaf)),
                'min_hessian': float(min_hessian),
                'reg_alpha': max(min(reg_alpha, 1), 0),
                'reg_lambda': max(min(reg_lambda, 10), 0),
                'early_stopping_rounds': int(round(early_stopping_rounds)),
                'bagging_seed': 42,
            }

        fit_params = {
            "num_boost_round": 1000000,
            "verbose_eval": 100,
        }
        score = train_lgb(params, fit_params, X, y, cv, use_custom_loss=True, get_score=True, drop_when_train=exclude_features, categorical_features=cat_cols)
        print(f"Score: {score}")
        return -score #Maximize

    var_params = {
                'num_leaves': (1000, 1500),
                'feature_fraction': (0.4, 1.0),
                'bagging_fraction': (0.5, 0.9),
                'bagging_freq': (1, 50),
                'learning_rate': (0.1, 0.5),
                'min_data_in_leaf': (10, 100),
                'min_hessian': (1e-4, 1e-1),
                'reg_alpha': (0.0 , 0.7),
                'reg_lambda': (0.0 , 10),
                'early_stopping_rounds': (20, 150),
                }

    lgbBO = BayesianOptimization(lgb_eval, var_params, random_state=42)
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    # return best parameters
    return lgbBO

opt_params = bayes_parameter_opt_lgb(X_train, y_train, cv)

with open('bayes_ope_result.txt', 'w') as handle:
    print(opt_params.max, file=handle)
    print("\n==========\n", file=handle)
    try:
        for key in opt_params.res:
            print(f"{key}: {opt_params.res[key]}", file=handle)
    except:        
        print(opt_params.res, file=handle)