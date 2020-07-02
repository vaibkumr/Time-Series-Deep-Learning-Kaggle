import gc
import os
import sys
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import register_matplotlib_converters
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_tweedie_deviance

import tqdm
import logging
import sys
import pprint
import uuid
import datetime
from utils import *
import make_data
from load_data import *
from losses import *
from sklearn.model_selection import GroupKFold, KFold


# 2011-01-29 ~ 2016-04-24 : d_1    ~ d_1913
# 2016-04-25 ~ 2016-05-22 : d_1914 ~ d_1941 (public)
# 2016-05-23 ~ 2016-06-19 : d_1942 ~ d_1969 (private)


warnings.filterwarnings("ignore")
seed_everything()
NUM_ITEMS = 30490
DAYS_PRED = 28
YEARS = 1
weights = 0 #global var
weights_test = 0 #global var
roll_mat_csr = 0 #global var
sw = 0 #global var
hess = None #global var

exclude_features = [
            'id',
            'wm_yr_wk',
            'date',
            'demand',
            'part',
            'd',
            # 'wts',
        ]

# 326.8MB

def load_rollup_matrix():
    global roll_mat_csr
    global sw
    roll_mat_df = pd.read_pickle('data/roll_mat_df.pkl')
    roll_mat_csr = csr_matrix(roll_mat_df.values)
    del roll_mat_df; gc.collect() 
    sw_df = pd.read_pickle('data/sw_df.pkl')
    sw = sw_df.sw.values
    # sw = sw/np.sum(sw)

def rollup(v):
    global roll_mat_csr
    return roll_mat_csr*v 


def wrmsse_obj(y_pred, y_true):
    y_true = y_true.get_label()
    grad = get_grad_no_weights(y_pred, y_true, roll_mat_csr)
    hess = 2*np.ones(len(y_pred))
    factor = 1.5
    grad = np.where(y_true==0, grad, factor*grad) #Penalize zeros
    return grad, hess
 

# def wrmsse_obj(y_pred, y_true):
#     y_true = y_true.get_label()
#     grad = get_grad(y_pred, y_true, roll_mat_csr, sw)
#     # hess = 2*np.ones(len(y_pred)) #Wrong hessian
#     hess = get_sudo_wrmsse_hess(l=len(y_true))
#     # print(grad.sum())
#     factor = 1.15
#     grad = np.where(y_true==0, grad*factor, grad) #Penalize zeros
#     return grad, hess

def custom_rolled_tweeedie(y_pred, y_true):
    y_true = y_true.get_label()
    rho=1.5
    grad = get_grad_tweedie(y_true, y_pred, roll_mat_csr, rho)
    # hess = tweedie_hessian(y_true, y_pred, rho)
    hess = 2*np.ones(len(y_pred)) #Wrong hessian
    # print(grad.sum())
    return grad, hess    

# def custom_asymmetric_train(y_pred, y_true):
#     y_true = y_true.get_label()
#     residual = (y_true - y_pred).astype("float")
#     grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)
#     hess = np.where(residual < 0, 2, 2 * 1.15)
    
#     print(np.sum(y_pred))

#     return grad, hess


# define custom evaluation metric
#WORKING:-
# def custom_asymmetric_valid(y_pred, y_true):
#     y_true = y_true.get_label()
#     residual = ((y_true - y_pred).astype("float").reshape(-1, NUM_ITEMS) * roll_mat_csr.T).flatten()
#     # loss = np.where(residual < 0, (residual ** 2) , (residual ** 2) * 1.15) 
#     # return "custom_asymmetric_eval", np.mean(loss), False
#     return "custom_asymmetric_eval", np.sqrt(np.mean(np.square(residual)))/12, False

# def custom_asymmetric_valid(y_pred, y_true):
#     y_true = y_true.get_label()
#     residual = ((y_true - y_pred).astype("float").reshape(-1, NUM_ITEMS) * roll_mat_csr.T).flatten()
#     loss = np.sqrt(np.mean(residual**2))
#     return "custom_asymmetric_eval", loss, False

def wrmsse_val(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float").reshape(NUM_ITEMS, -1)
    loss = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(rollup(residual))
                            ,axis=1))*sw)/12 
    return "wrmsse", loss, False    


def tweedie_eval(y_pred, y_true):
    y_true = y_true.get_label()
    y_true = rollup(y_true.reshape(NUM_ITEMS, -1)).flatten()
    y_pred = rollup(y_pred.reshape(NUM_ITEMS, -1)).flatten()
    loss = mean_tweedie_deviance(y_true, y_pred, power=1.5)
    return "tweedie_eval", loss, False       

def get_sudo_wrmsse_hess(l):
    global hess
    if hess is not None:
        return hess
    hess = np.ones(l).reshape(NUM_ITEMS, -1)
    hess = rollup(hess).T*sw
    hess = (hess*roll_mat_csr).flatten()
    return hess


def weighted_rmse(y_pred, y_true):
    global weights_test
    global weights

    if len(y_pred)==len(weights_test):
        return 'weighted rmse', np.sqrt(np.sum(weights_test*np.square(y_pred-y_true.get_label()))/np.sum(weights_test)), False    
    return 'weighted rmse', np.sqrt(np.sum(weights*np.square(y_pred-y_true.get_label()))/np.sum(weights)), False    

def train_lgb(bst_params, fit_params, X, y, cv,
              drop_when_train=None, categorical_features=[],
              use_custom_loss=False, get_score=False, random_fold=False,
              nf=1,
              ):
    
    global weights
    global weights_test

    models = []
    score = 0

    logger = logging.getLogger("global")

    if drop_when_train is None:
        drop_when_train = []

    dates = X.date.unique()
    kf = KFold(nf, shuffle=True)

    for idx_fold, (trn_idx, val_idx) in enumerate(kf.split(dates)):
        print(f"\n----- Fold: ({idx_fold + 1} / {nf}) -----\n")

        train_dates = dates[trn_idx]
        val_dates = dates[val_idx] 
        X_train = X[X['date'].isin(train_dates)]
        X_val = X[X['date'].isin(val_dates)]

        train_set = lgb.Dataset(X_train.drop(drop_when_train, axis=1).values,
                                label=X_train.demand.values, 
                                categorical_feature=categorical_features)
        train_set.raw_data = None                                
        print("Created train set")                        
        
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis=1).values,
                                label=X_val.demand.values, 
                                categorical_feature=categorical_features)
        val_set.raw_data = None       
        print("Created valid set")                        
        gc.collect()                         

        if use_custom_loss:
            print(f"Using custom loss!!")
            feval = None
            fobj = wrmsse_obj
        else:
            print(f"Not using custom loss.")
            fobj = None
            feval = None    
      
        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            fobj=fobj,
            feval=feval,
            # learning_rates=lambda iter: 0.0007 if iter>400 else 0.00007,
            **fit_params,
            )

        if get_score:
            y_preds = model.predict(X_val.drop(drop_when_train,
                                    axis=1).values)
            score += rmse(X_val.demand.values, y_preds)/nf
        models.append(model)
        

        del X_train, X_val
        gc.collect()


        
    # for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y, group)):

    #     print(f"\n----- Fold: ({idx_fold + 1} / {nf}) -----\n")

    #     # X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
    #     # y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]

    #     train_set = lgb.Dataset(X.iloc[idx_trn].drop(drop_when_train, axis=1).values,
    #                             label=y.iloc[idx_trn].values, 
    #                             categorical_feature=categorical_features)
    #     train_set.raw_data = None                                
    #     print("Created train set")                        
        
    #     val_set = lgb.Dataset(X.iloc[idx_val].drop(drop_when_train, axis=1).values,
    #                             label=y.iloc[idx_val].values, 
    #                             categorical_feature=categorical_features)
    #     val_set.raw_data = None       
    #     gc.collect()                         

    #     print("Created valid set")                        

    #     if use_custom_loss:
    #         print(f"Using custom loss!!")
            
    #         # weights = X.iloc[idx_trn]['wts']  
    #         # weights_test = X.iloc[idx_val]['wts'] 

    #         feval = custom_asymmetric_valid
    #         fobj = custom_asymmetric_train

    #     else:
    #         print(f"Not using custom loss.")
    #         fobj = None
    #         feval = None    
      
    #     model = lgb.train(
    #         bst_params,
    #         train_set,
    #         valid_sets=[train_set, val_set],
    #         valid_names=["train", "valid"],
    #         fobj=fobj,
    #         feval=feval,
    #         **fit_params,
    #         )

    #     if get_score:
    #         y_preds = model.predict(X.iloc[idx_val].drop(drop_when_train,
    #                                 axis=1).values)
    #         score += rmse(y.iloc[idx_val].values, y_preds)/nf
    #     models.append(model)
        

    #     del idx_trn, idx_val
    #     gc.collect()

    return score if get_score else models

def predict_and_f_importance(models, X_test, n_splits=5, imp_type = "gain",
                            exclude_features=None, combine_by='mean'):
    if exclude_features is None:
        exclude_features = []

    X_test = X_test.drop(exclude_features, axis=1)
    print(X_test.shape)
    importances = np.zeros(X_test.shape[1])
    preds = pd.DataFrame()


    for i, model in enumerate(models):
        preds[f'model_{i}'] = model.predict(X_test)
        importances += model.feature_importance(imp_type)

    weights = []
    if combine_by == 'corr':
        for i, col in enumerate(preds.columns):
            cur = preds[col]
            others = preds.loc[:, preds.columns != col].mean(axis=1)
            corr = np.corrcoef(cur, others)[0][1]
            weights.append(1/corr)
    else:
        weights = [1 for x in range(len(models))]

    finalpreds = np.zeros(X_test.shape[0])
    for w, col in zip(weights, preds.columns):
        finalpreds += w*preds[col]

    finalpreds /= sum(weights)
    importances /= n_splits

    return finalpreds, importances

def predict_recursive(models, X_test, n_splits=5, imp_type = "gain",
                            exclude_features=None, combine_by='mean'):
    if exclude_features is None:
        exclude_features = []

    # X_test = X_test.drop(exclude_features, axis=1)
    print(X_test.shape)
    preds = pd.DataFrame()

    fday = datetime.datetime(2016,4, 25)
    max_lags = int(28*2)+2
    alpha = 1

    for tdelta in tqdm.tqdm(range(0, 28)):
    # for tdelta in tqdm.tqdm(range(0, 1)):
        day = fday + datetime.timedelta(days=tdelta)
        tst = X_test[(X_test.date >= day - datetime.timedelta(days=max_lags)) & (X_test.date <= day)].copy()
        tst = make_data.add_demand_features_minimal(tst, recursive=True)
        tst = tst.drop(exclude_features, axis=1).loc[tst.date == day]
        for model in models:
            X_test.loc[X_test.date == day, "demand"] += alpha*model.predict(tst)/len(models) #alpha is the **magic** factor, I dont trust it. So I just set it to 1.

    return X_test[X_test.date>=fday]['demand'].values

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_submission(test, fname="submission.csv"):
    submission = pd.read_csv(f"data/sample_submission.csv")
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    evals = submission[submission["id"].str.endswith("evaluation")]
    vals = submission[["id"]].merge(preds, how="inner", on="id")
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv(fname, index=False)
    return final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M5 accuracy')

    parser.add_argument("-data", "--d", type=str, 
                        default="/home/timetraveller/Work/m5data/features.hdf")
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
                                                                                                                                                                                                                                               

  
    
    args = parser.parse_args()
    # project_name = str(uuid.uuid4())
    

    num_boost_round = args.nbr
    submit_kaggle = args.ks
    just_predict = args.j
    path = args.d
    logfname = args.l
    outdir = args.o
    modelname = args.m
    nf = args.nf
    kaggle_message = args.km
    target_encoding = args.te
    random_fold = args.rf
    params = args.pr
    dropna = args.dn
    use_custom_loss = args.cl
    early_stopping_rounds = args.esr
    recursive = args.rec 
    calc_wrmsse = args.wrmsse



    print(f"submit_kaggle: {submit_kaggle}")
    print(f"use_custom_loss: {use_custom_loss}")
    print(f"recursive: {recursive}")
    print(f"random_fold: {random_fold}")
    print(f"calc_wrmsse: {calc_wrmsse}")
    print(f"num_folds: {nf}")

    
    project_name = str(datetime.datetime.today()).replace(' ', '_')
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

    # data = pd.read_hdf(path).pipe(reduce_mem_usage)
    data = pd.read_hdf(path)
    
    if dropna:
        bef = data.shape[0]
        print(f"#Rows before: {bef}")
        data.dropna(inplace = True)
        print(f"#Rows after: {data.shape[0]} ({(bef-data.shape[0])/bef}% reduction)")

    if use_custom_loss:
        load_rollup_matrix()   

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
    ]
    cat_cols = [] #Memory issues

    train_end = datetime.datetime(2016, 4, 24)
    # train_start = datetime.datetime(2014, 2, 24)
    train_start = datetime.datetime(2014, 4, 24)
    test_start = train_end - datetime.timedelta(days=28)

    # mask = data["date"] <= "2016-04-24"  #19727030
    # mask2 = data["date"] >= "2014-04-24" #FUCK LARGE DATA AND MY SMALL RAM

    # train_mask = data["date"] <= test_start 
    train_mask = (data["date"] <= test_start) & (data["date"] > train_start) #FUCK LARGE DATA AND MY SMALL RAM
    val_mask = (data["date"] <= train_end) & (data["date"] > test_start)
    test_mask = data["date"] > train_end



    X_train = data[train_mask].reset_index(drop=True)
    y_train = data[train_mask]["demand"].reset_index(drop=True)


    X_val = data[val_mask].reset_index(drop=True) #For oof WRMSSE
    
    X_test = data[test_mask].reset_index(drop=True)
    id_date = data[test_mask][["id", "date"]].reset_index(drop=True)
    

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")   
    logger.info(list(X_train.columns))

    del data, train_mask, val_mask, test_mask
    # del X_test, y_train
    gc.collect()
    

    if random_fold:
        cv = GroupKFold(nf)
    else:   
        cv = TimeSeriesCV(n_splits=nf, predict_days=28, date_col="date")

    if just_predict:
        model_pth = "something"
        models = [load_model(model_pth)[-1]]

        # finalpreds, importances = predict_and_f_importance(models, X_test,
        #                         n_splits=nf, exclude_features=exclude_features)

        finalpreds = predict_recursive(models, pd.concat([X_train, X_test]),
                                n_splits=nf, exclude_features=exclude_features)                        
        make_submission(id_date.assign(demand=finalpreds))
        sys.exit()


    if params == 'p1':
        # bst_params = {
        #             'boosting_type': 'gbdt',
        #             'subsample': 0.70,
        #             'subsample_freq': 1,
        #             'learning_rate': 0.09,
        #             'num_leaves': 2**11-1,
        #             'min_data_in_leaf': 2**12-1,
        #             'feature_fraction': 0.5,
        #             'max_bin': 120,
        #             'boost_from_average': False,
        #             'verbose': -1,
        # }   
        
        bst_params = {
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'seed': 42,
            'learning_rate': 0.00007,
            # 'num_leaves': 200,
            'num_leaves': 50,
            # 'num_leaves': 15,
            # 'max_bin': 16,
            'max_bin': 20,
            'bagging_fraction': 0.85,
            'bagging_freq': 1, 
            'colsample_bytree': 0.65,
            'colsample_bynode': 0.65,
            'min_data_per_leaf': 45,
            'lambda_l1': 0.75,
            'lambda_l2': 0.75,
            'metric':'rmse',
        }

        if not use_custom_loss:
            bst_params['objective']='tweedie'
            bst_params['tweedie_variance_power']=1.1
            bst_params['metric']='rmse'

        fit_params = {
                'num_boost_round': num_boost_round,
                'verbose_eval': 50,
                'early_stopping_rounds': early_stopping_rounds,
        } 
    else:    

        bst_params = {
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'seed': 42,
            'learning_rate': 0.05,
            # 'learning_rate': 0.25,
            'num_leaves': 200,
            'bagging_fraction': 0.85,
            'bagging_freq': 1, 
            'colsample_bytree': 0.75,
            'colsample_bynode': 0.75,
            'min_data_per_leaf': 35,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'metric':'rmse',
        }
        
        if not use_custom_loss:
            bst_params['objective']='tweedie'
            bst_params['metric']='rmse'


        fit_params = {
                'num_boost_round': num_boost_round,
                # 'verbose_eval': 100,
                'verbose_eval': 50,
                'early_stopping_rounds': early_stopping_rounds,
        }

    # bst_params = clean_params(bst_params)
    logger.info(bst_params)
    logger.info(fit_params)

    day_col = 'd'
    logger.info(f"Custom loss: [{use_custom_loss}]")
    
    models = train_lgb(
        bst_params, fit_params, X_train, y_train, cv, 
        drop_when_train=exclude_features, categorical_features=cat_cols,
        use_custom_loss=use_custom_loss,
        random_fold=random_fold, nf=nf,
    )


    model_pth = os.path.join(outdir, project_name, modelname)

    save_model(models, model_pth)
    logger.info(f"Model saved in [{model_pth}]")

    if calc_wrmsse:
        wrmsse, wrmsse_matrix = get_wrmsse(X_val, models, exclude_features)
        logger.info(f"WRMSSE: {wrmsse}")
        
        matrix_pth = os.path.join(outdir, project_name, "wrmsse_matrix.csv")
        pd.DataFrame(wrmsse_matrix).to_csv(matrix_pth, index=False)

        logger.info(f"WRMSSE Matrix saved in {matrix_pth}")

    if recursive:
        idx = int(NUM_ITEMS * 30) 
        finalpreds = predict_recursive(models, pd.concat([X_train.iloc[-idx:], X_test]),
                                n_splits=nf, exclude_features=exclude_features) 
    else:    
        finalpreds, importances = predict_and_f_importance(models, X_test,
                                n_splits=nf, exclude_features=exclude_features)
    
    subfname="submission.csv"    
    
    del X_train, y_train
    gc.collect()
    
    subfname = os.path.join(outdir, project_name, f"{project_name}-submission.csv")

    make_submission(id_date.assign(demand=finalpreds), subfname)
    logger.info(f"Submission file created: {subfname}")

    if submit_kaggle:
        os.system(f"kaggle competitions submit -f {subfname} -m \"{kaggle_message}\" -c m5-forecasting-accuracy")  
        logger.info(f"Submitted to kaggle!")
      

# [50]	train's rmse: 3.11182	train's custom_asymmetric_eval: 0.350369	valid's rmse: 2.99247	valid's custom_asymmetric_eval: 0.328893
# [100]	train's rmse: 2.76676	train's custom_asymmetric_eval: 0.311623	valid's rmse: 2.66218	valid's custom_asymmetric_eval: 0.289507
# [150]	train's rmse: 2.53627	train's custom_asymmetric_eval: 0.286832	valid's rmse: 2.44716	valid's custom_asymmetric_eval: 0.266578
# [200]	train's rmse: 2.39405	train's custom_asymmetric_eval: 0.272246	valid's rmse: 2.31933	valid's custom_asymmetric_eval: 0.253433
# [250]	train's rmse: 2.30883	train's custom_asymmetric_eval: 0.2639	valid's rmse: 2.24624	valid's custom_asymmetric_eval: 0.245816
# [300]	train's rmse: 2.25743	train's custom_asymmetric_eval: 0.259008	valid's rmse: 2.205	valid's custom_asymmetric_eval: 0.241774
# [350]	train's rmse: 2.22452	train's custom_asymmetric_eval: 0.255886	valid's rmse: 2.18047	valid's custom_asymmetric_eval: 0.239393
# [400]	train's rmse: 2.20204	train's custom_asymmetric_eval: 0.253824	valid's rmse: 2.16549	valid's custom_asymmetric_eval: 0.238016
# [450]	train's rmse: 2.18481	train's custom_asymmetric_eval: 0.252227	valid's rmse: 2.15477	valid's custom_asymmetric_eval: 0.236979
# [500]	train's rmse: 2.17178	train's custom_asymmetric_eval: 0.251014	valid's rmse: 2.14669	valid's custom_asymmetric_eval: 0.236422



