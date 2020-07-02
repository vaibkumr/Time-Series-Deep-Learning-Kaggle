import gc
import os
import torch
import pickle
import argparse
import lightgbm as lgb
import logging
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta

from utils import *


NUM_ITEMS = 30490
roll_mat_csr = 0

def load_roll_mat():
    global roll_mat_csr
    roll_mat_df = pd.read_pickle('data/roll_mat_df.pkl')
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df; gc.collect() 

def rollup(v):
    return roll_mat_csr*v 

def get_grad(y_pred, y_true, roll_mat_csr):
    """autograd based gradient calculation for custom loss"""
    roll_mat_coo = roll_mat_csr.tocoo()
    rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
                              torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)

    y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
    y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

    residual = y_true_t-y_pred_t

    loss = 1e2*torch.sum(torch.sqrt(
                torch.mean((
                    torch.mm(rollTensor, residual)**2), axis=1)+1e-8 #to prevent nans in backprop as log0 is not defined
                ))     

    loss.backward()

    grad = y_pred_t.grad.detach().numpy().flatten()
    return grad     

def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    grad = get_grad(y_pred, y_true, roll_mat_csr)
    hess = 2*np.ones(len(y_pred))
    return grad, hess

def custom_asymmetric_valid(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float").reshape(NUM_ITEMS, -1)
    loss = np.mean(
                np.sqrt(
                    np.mean(
                        np.square(rollup(residual))
                            ,axis=1)))/12 
    return "custom_asymmetric_eval", loss, False       


CAL_DTYPES={
            "event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
            "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
            "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' 
           }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday

def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("data/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("data/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("data/sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, how="left", on= "d", copy = False)
    dt = dt.merge(prices, how="left", on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return dt

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        rate = 0.85
        weights = np.array([(rate**i) for i in range(win)])
        sum_w = np.sum(weights)
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
            # dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).apply(lambda x: np.sum(weights*x)/sum_w))

    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")    


FIRST_DAY = 520 #521 is a prime 

def load_index(index_data_path):
    """I am loading index and not calculating it on fly because of RAM issues"""
    with open(index_data_path, "rb") as handle:
        idx = pickle.load(handle)
    return idx    

def get_params(use_custom_loss=True, params_name="p1", num_boost_round=1500, classify=True):
    params = {
        "force_row_wise" : True,
        "learning_rate" : 0.1,
        "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,
        'num_iterations' : num_boost_round,
        'num_leaves': 128,
        "min_data_in_leaf": 104,
     }

    if classify:
        params['objective'] = "cross_entropy"
        params['metric'] = ["auc", "cross_entropy"]

    if not use_custom_loss and not classify:
         params['objective'] = "tweedie"
         params['metric'] = "rmse"

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M5 accuracy')

    parser.add_argument("-data", "--d", type=str, 
                        default="/home/timetraveller/Work/m5data/minimal.hdf")
    parser.add_argument("-index_data", "--idd", type=str, 
                        default="/home/timetraveller/Work/m5data/idx.pkl")                        
    parser.add_argument("-outdir", "--o", type=str, 
                        default="/home/timetraveller/Work/m5data/output")
    parser.add_argument("-logfname", "--l", type=str, 
                        default="m5acc.log")   
    parser.add_argument("-modelname", "--m", type=str, 
                        default="model.pkl")  
    parser.add_argument("-num_boost_round", "--nbr", type=int, 
                        default=5000)   
    parser.add_argument("-num_folds", "--nf", type=int, 
                        default=5)     
    parser.add_argument("-early_stopping_rounds", "--esr", type=int, 
                        default=50)  
    parser.add_argument("-params", "--pr", type=str,
                        default="p2")    
    parser.add_argument("-kaggle_message", "--km", type=str,
                        default="gambatte!")      
    
    #Boolean args:-
    parser.add_argument("-target_encoding", "--te", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("-justpredict", "--j", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("-submit_kaggle", "--ks", type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("-random_fold", "--rf", type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("-dropna", "--dn", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("-use_custom_loss", "--cl", type=str2bool, nargs='?',
                        const=True, default=False)        
    parser.add_argument("-recursive", "--rec", type=str2bool, nargs='?',
                        const=True, default=False)    
    parser.add_argument("-calc_wrmsse", "--wrmsse", type=str2bool, nargs='?',
                        const=True, default=False)  
    parser.add_argument("-classify", "--csf", type=str2bool, nargs='?',
                        const=True, default=False)                                 

    args = parser.parse_args()

    num_boost_round = args.nbr
    submit_kaggle = args.ks
    just_predict = args.j
    path = args.d
    index_data_path = args.idd
    logfname = args.l
    outdir = args.o
    modelname = args.m
    nf = args.nf
    kaggle_message = args.km
    target_encoding = args.te
    random_fold = args.rf
    params_name = args.pr
    dropna = args.dn
    use_custom_loss = args.cl
    early_stopping_rounds = args.esr
    recursive = args.rec 
    calc_wrmsse = args.wrmsse
    classify = args.csf

    print(f"submit_kaggle: {submit_kaggle}")
    print(f"use_custom_loss: {use_custom_loss}")
    print(f"recursive: {recursive}")
    print(f"random_fold: {random_fold}")
    print(f"calc_wrmsse: {calc_wrmsse}")
    print(f"num_folds: {nf}")        

    project_name = str(datetime.today()).replace(' ', '_')
    make_dir(os.path.join(outdir, project_name))

    logfile = os.path.join(outdir, project_name, logfname)   
    stdoutfile = os.path.join(outdir, project_name, "stdout.txt")   
        

    fhandler = logging.FileHandler(logfile)
    fhandler.setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO,
            )
                                
    logger = logging.getLogger("global")
    logger.addHandler(fhandler)
    logger.info(f"Training on: {args.d}")
    logger.info(f"submit_kaggle: {submit_kaggle}")
    logger.info(f"use_custom_loss: {use_custom_loss}")
    logger.info(f"recursive: {recursive}")
    logger.info(f"random_fold: {random_fold}")
    logger.info(f"num_folds: {nf}")

    df = pd.read_hdf(path)         

    # cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
    cat_feats = []
    useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]

    idx = load_index(index_data_path)

    train_inds = idx['train'].values
    fake_valid_inds = idx['val'].values

    print(len(train_inds))
    print(len(fake_valid_inds))

    X_train = df[train_cols]
    y_train = df["sales"]


    if classify:
        modify = lambda x: 1 if x == 0 else 0
    else:
        modify = lambda x: x    

    train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds].apply(modify), 
                        categorical_feature=cat_feats)

    fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds].apply(modify),
                        categorical_feature=cat_feats)

    print(f"Datasets made.")                      

    del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()
    df = pd.DataFrame()
    del df; gc.collect()

    params = get_params(use_custom_loss, params_name, num_boost_round, classify)

    feval = None
    fobj = None

    if use_custom_loss:
        feval = custom_asymmetric_valid
        fobj = custom_asymmetric_train

    m_lgb = lgb.train(params, train_data, 
                    valid_sets = [train_data, fake_valid_data], 
                    verbose_eval=100,
                    fobj=fobj,
                    feval=feval
                    ) 

    model_pth = os.path.join(outdir, project_name, modelname)

    save_model(models, model_pth)
    logger.info(f"Model saved in [{model_pth}]")

    
    # Make output
    # alphas = [1.035, 1.03, 1.025]
    alphas = [1]
    weights = [1/len(alphas)]*len(alphas)
    sub = 0.

    for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

        te = create_dt(False)
        cols = [f"F{i}" for i in range(1,29)]

        for tdelta in range(0, 28):
            day = fday + timedelta(days=tdelta)
            print(icount, day)
            tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
            create_fea(tst)
            tst = tst.loc[tst.date == day , train_cols]
            te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev

        te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
        te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
        te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
        te_sub.fillna(0., inplace = True)
        te_sub.sort_values("id", inplace = True)
        te_sub.reset_index(drop=True, inplace = True)
        te_sub.to_csv(f"submission_{icount}.csv",index=False)
        if icount == 0 :
            sub = te_sub
            sub[cols] *= weight
        else:
            sub[cols] += te_sub[cols]*weight
        print(icount, alpha, weight)

    sub2 = sub.copy()
    sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
    sub = pd.concat([sub, sub2], axis=0, sort=False)
    subfname = os.path.join(outdir, project_name, f"{project_name}-submission.csv")
    sub.to_csv(subfname,index=False)
              

