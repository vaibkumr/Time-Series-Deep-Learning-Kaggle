import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

NUM_ITEMS = 30490

def rollup(v, roll_mat_csr):
    '''
    v - np.array of size (30490 rows, n day columns)
    v_rolledup - array of size (n, 42840)
    '''
    return roll_mat_csr*v #(v.T*roll_mat_csr.T).T


def wrmsse(preds, y_true, score_only=True, npy=True, roll_mat_csr=None, sw=None):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''
    if roll_mat_csr is None:
        roll_mat_df = pd.read_pickle('data/roll_mat_df.pkl')
        roll_index = roll_mat_df.index
        roll_mat_csr = csr_matrix(roll_mat_df.values)
        del roll_mat_df; gc.collect()

    if sw is None:
        sw_df = pd.read_pickle('data/sw_df.pkl')
        s = sw_df.s.values
        w = sw_df.w.values
        sw = sw_df.sw.values

    if not npy:
        preds = preds.values
        y_true = y_true.values
    
    if score_only:
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(rollup(preds-y_true, roll_mat_csr))
                            ,axis=1)) * sw)/12 
    else: 
        score_matrix = (np.square(rollup(preds-y_true, roll_mat_csr)) * np.square(w)[:, None])/ s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix,axis=1)))/12 
        return score, score_matrix


def get_wrmsse(X_test, models, exclude_features):
    print(f"Calculating WRMSSE...")
    y_true = X_test.demand.values    
    y_pred = np.zeros_like(y_true).astype(np.float64)
    for model in tqdm(models):
        y_pred += model.predict(X_test.drop(exclude_features, axis=1))/len(models)        
    return wrmsse(y_pred.reshape(NUM_ITEMS, -1), 
                  y_true.reshape(NUM_ITEMS, -1), 
                  npy=True)            

def get_grad_no_weights(y_pred, y_true, roll_mat_csr):
    """autograd based gradient calculation for custom loss"""
    roll_mat_coo = roll_mat_csr.tocoo()
    rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
                              torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)

    y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
    y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

    # residual = y_true_t-y_pred_t
    y_true_t_rolled = torch.mm(rollTensor, y_true_t)
    y_pred_t_rolled = torch.mm(rollTensor, y_pred_t)
    # residual = y_pred_t_rolled - y_true_t_rolled 
    residual = torch.log1p(y_pred_t_rolled) - torch.log1p(y_true_t_rolled) 

    # loss = torch.sqrt(
    #             torch.mean(
    #                 (torch.mm(
    #                     rollTensor, residual)**2
    #             ))) #-219.82277

    # loss = 1e2*torch.sum(torch.sqrt(
    #             torch.mean((
    #                 torch.mm(rollTensor, residual)**2), axis=1)+1e-8 #to prevent nans in backprop as log0 is not defined
    #             )) #-325249.53 

    loss = 9*1e2*torch.sum(torch.sqrt(
                torch.mean((
                    residual**2), axis=1)+1e-8 #to prevent nans in backprop as log0 is not defined
                )) #-325249.53                        

    # loss = torch.sum(residual**2) #-32005542.0 - WORKS - This is what loss should be in scale with
    # loss = 1e2*torch.mean(torch.mm(rollTensor, residual)**2) #-10065.081
    loss.backward()
    # print(loss)

    grad = y_pred_t.grad.detach().numpy().flatten()
    print(grad.sum())
    return grad      

# Scales:
# -32133866.0 LGB residuals
# -1452104.2
# -1414693.2
# -1378524.8
# -1343508.1
# -1309641.8
# -1276749.2
# -1244916.6
# -1214047.9
# -1184132.8
# -1155058.1

# [50]	train's rmse: 3.05002	valid's rmse: 2.98111

# [50]	train's rmse: 3.3211	valid's rmse: 3.24301
# [50]	train's rmse: 3.18068	valid's rmse: 3.10543

# [50]	train's rmse: 3.4455	valid's rmse: 3.36632
# [100]	train's rmse: 3.35159	valid's rmse: 3.27364


def get_grad(y_pred, y_true, roll_mat_csr, sw):
    """autograd based gradient calculation for custom loss"""
    roll_mat_coo = roll_mat_csr.tocoo()
    rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
                              torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)
    sw_t = torch.FloatTensor(sw)                                                    

    y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
    y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

    residual = y_true_t-y_pred_t
    # -32218056.0
    # -36301877

    loss = 1e7*torch.sum(torch.sqrt(
                torch.mean((
                    torch.mm(rollTensor, residual)**2), axis=1)+1e-8 #to prevent nans in backprop as log0 is not defined
                )*sw_t)/12         

    # loss = torch.sum(residual**2) #-32005542.0 - WORKS - This is what loss should be in scale with
    # loss = 1e2*torch.mean(torch.mm(rollTensor, residual)**2) #-10065.081
    loss.backward()
    # print(loss)

    grad = y_pred_t.grad.detach().numpy().flatten()
    return grad   

# def get_grad_tweedie(y_true, y_pred, roll_mat_csr, rho=1.5):
#     """
#     partially working
#     https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances
#     https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/metric/regression_metric.hpp#L300-L318
#     https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/objective/regression_objective.hpp#L702-L732
#     """
#     roll_mat_coo = roll_mat_csr.tocoo()
#     rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
#                               torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)

#     eps = 1e-10

#     factor = 1
#     if y_pred.sum() <= eps:
#         print("wtahsd!!")
#         factor = 1e19
#         # y_pred = np.random.rand(len(y_pred))

#     y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
#     y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

#     rolled_y_true = torch.mm(rollTensor, y_true_t).flatten() + eps
#     rolled_y_pred = torch.mm(rollTensor, y_pred_t).flatten() + eps

#     # if y_pred.sum() < 0.5:
#     #     a = 0 
#     # else:
#     #     a = ( rolled_y_true * torch.pow(rolled_y_pred, (1-rho)) ) / (1-rho) 
#     # b = torch.pow(rolled_y_pred, (2-rho))  / (2-rho) 
#     # # c = torch.pow(rolled_y_true, (2-rho)) / ((1-rho)*(2-rho))
#     # tweedie = -a + b

#     log_y = torch.log(F.relu(rolled_y_pred)+eps) #Sometimes there are negative values which give nan
#     a = rolled_y_true * torch.exp(log_y*(1-rho)) / (1-rho)
#     b = torch.exp(log_y*(2-rho)) / (2-rho)
#     tweedie = -a + b

#     # y = torch.log(y_pred_t)
#     # x = torch.exp(y*(1-rho))
#     # a = y_true_t * x / (1-rho)
#     # b = torch.exp(torch.log(y_pred_t)*(2-rho)) / (2-rho)
#     # tweedie = -a + b
    
#     # loss = torch.mean(tweedie)
#     loss = torch.sum(tweedie)/factor
    
#     # loss = 1e7*torch.sum(torch.sqrt(
#     #             torch.mean((
#     #                 tweedie), axis=1)+1e-8 #to prevent nans in backprop as log0 is not defined
#     #             ))/12       
    
    
#     loss.backward()
#     grad = y_pred_t.grad.detach().numpy().flatten()

#     debug = True
#     if debug:
#         print(f"Loss: {loss}")
#         with torch.no_grad():
#             print(f"a: {torch.mean(a)}")
#             print(f"b: {torch.mean(b)}")
#             print(f"log_y: {torch.mean(log_y)}")
#             print(f"y_true_t: {torch.mean(y_true_t)}")
#             print(f"y_pred_t: {torch.mean(y_pred_t)}")
#             print(f"min y_pred_t: {torch.min(y_pred_t)}")
#             print(grad.sum())

#     return grad      


def get_grad_tweedie(y_true, y_pred, roll_mat_csr, rho=1.5):
    """
    fixing above, workin progress
    https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances
    https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/metric/regression_metric.hpp#L300-L318
    https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/objective/regression_objective.hpp#L702-L732
    """
    roll_mat_coo = roll_mat_csr.tocoo()
    rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
                              torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)

    eps = 1e-10

    factor = 1
    if y_pred.sum() <= eps:
        print("wtahsd!!")
        factor = 1e19
        # y_pred = np.random.rand(len(y_pred))
    else:
        # y_pred = np.where(y_pred<0, eps, y_pred)  #Filter 0 and negative values 
        y_pred = np.abs(y_pred)

    y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
    y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

    rolled_y_true = torch.mm(rollTensor, y_true_t).flatten() 
    rolled_y_pred = torch.mm(rollTensor, y_pred_t).flatten() + eps


    # # log_y = torch.log(F.relu(rolled_y_pred)+eps) #Sometimes there are negative values which give nan
    # log_y = torch.log(rolled_y_pred) #Sometimes there are negative values which give nan
    # a = rolled_y_true * torch.exp(log_y*(1-rho)) / (1-rho)
    # b = torch.exp(log_y*(2-rho)) / (2-rho)
    # tweedie = -a + b

    a = rolled_y_true * torch.pow(rolled_y_pred, (1-rho))  / (1-rho) 
    b = torch.pow(rolled_y_pred, (2-rho))  / (2-rho) 
    tweedie = -a + b
    loss = torch.sum(tweedie)/factor
    
    # 87578120.0
    # 421865216.0

    loss.backward()
    grad = y_pred_t.grad.detach().numpy().flatten()

    # #Gradient clipping
    LARGE = 180
    grad = np.where(grad>=10, 10, grad)
    grad = np.where(grad<=-LARGE, -LARGE, grad)

    debug = False
    if debug:
        print(f"Loss: {loss}")
        with torch.no_grad():
            print(f"a: {torch.mean(a)}")
            print(f"a min: {torch.min(a)}")
            print(f"b: {torch.mean(b)}")
            print(f"b min: {torch.min(b)}")
            # print(f"log_y: {torch.mean(log_y)}")
            # print(f"log_y_min: {torch.min(log_y)}")
            print(f"y_true_t: {torch.mean(y_true_t)}")
            print(f"y_pred_t: {torch.mean(y_pred_t)}")
            print(f"min y_pred_t: {torch.min(y_pred_t)}")
            print(f"min rolled_y_pred: {torch.min(y_pred_t)}")
            print(f"grad: {grad.mean()}")
            print(f"min grad: {grad.min()}")
            print(f"max grad: {grad.max()}")
            print(grad.sum())

    return grad  



# def get_grad_tweedie(y_true, y_pred, roll_mat_csr, rho=1.5):
#     """
#     autograd the integral of the derivative used by lightgbm
#     THIS FAILED. e^y_pred is soo large... how tf does it work for lightgbm?! what am I missing??
#     https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances
#     https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/metric/regression_metric.hpp#L300-L318
#     https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/objective/regression_objective.hpp#L702-L732
#     """
#     roll_mat_coo = roll_mat_csr.tocoo()
#     rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
#                               torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)

#     eps = 1e-10

#     factor = 1e11
#     factor2 = 1e7

#     # if y_pred.sum() <= eps:
#     #     print("wtahsd!!")
#     #     factor = 1e19
#     #     # y_pred = np.random.rand(len(y_pred))

#     y_pred = np.where(y_pred<0, eps, y_pred)  #Filter 0 and negative values 

#     y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
#     y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

#     rolled_y_true = torch.mm(rollTensor, y_true_t).flatten() + eps
#     rolled_y_pred = torch.mm(rollTensor, y_pred_t).flatten() + eps


#     # log_y = torch.log(F.relu(rolled_y_pred)+eps) #Sometimes there are negative values which give nan
#     # log_y = torch.log(rolled_y_pred) 
#     #Normalize
#     max_true = torch.max(rolled_y_true)
#     max_pred = torch.max(rolled_y_pred)

#     normal_y_true = rolled_y_true/max_true
#     normal_y_pred = rolled_y_pred/max_pred
    
#     a = normal_y_true * torch.exp(normal_y_pred*(1-rho)) / (1-rho)
#     b = torch.exp( normal_y_pred*(2-rho) ) / (2-rho) 

#     # a = rolled_y_true * torch.exp(rolled_y_pred*(1-rho)) / (1-rho)
#     # b = torch.exp( rolled_y_pred*(2-rho) ) / (2-rho) 
#     # # b becomes inf very quickly (anything above ~750), I wonder how does gradients[i] = static_cast<score_t>(-label_[i] * std::exp((1 - rho_) * score[i]) + std::exp((2 - rho_) * score[i])) workout
#     tweedie = -a + b

#     loss = torch.sum(tweedie)/factor2
    
    
#     loss.backward()
#     grad = y_pred_t.grad.detach().numpy().flatten()/factor

#     debug = True
#     if debug:
#         print(f"Loss: {loss}")
#         with torch.no_grad():
#             print(f"a: {torch.mean(a)}")
#             print(f"b: {torch.mean(b)}")
#             # print(f"log_y: {torch.mean(log_y)}")
#             print(f"y_true_t: {torch.mean(y_true_t)}")
#             print(f"y_pred_t: {torch.mean(y_pred_t)}")
#             print(f"min y_pred_t: {torch.min(y_pred_t)}")
#             print(f"max y_pred_t: {torch.max(y_pred_t)}")
#             print(f"rolled_y_pred: {torch.mean(rolled_y_pred)}")
#             print(f"min rolled_y_pred: {torch.min(rolled_y_pred)}")
#             print(f"max rolled_y_pred: {torch.max(rolled_y_pred)}")
#             print(grad.sum())

#     return grad  

# -82015855.000.0
# -4926122.5
# -32133866.0 LGB residuals (1e7)
# -3200416000000.0
# -909652460000.0


def tweedie_hessian(y_true, y_pred, rho=1.5):
    a = -y_true*(1-rho)*np.exp(1-rho)*y_pred
    b = (2-rho)*np.exp(2-rho)*y_pred
    return -a+b


# Scales:
# -32005542.0 mine
# -32133866.0 LGB residuals
# -23217492.0
# +854056600.0
# -325249


# 274.7243 loss
# 564742.3125 loss
# 13.1826 loss


def QLL(predicted, observed):
    p = torch.tensor(1.5)
    QLL = torch.pow(predicted, (-p))*(((predicted*observed)/(1-p)) - ((torch.pow(predicted, 2))/(2-p)))
    return QLL


def tweedieloss(predicted, observed):
    '''
    Custom loss fuction designed to minimize the deviance using stochastic gradient descent
    tweedie deviance from McCullagh 1983

    '''
    d = -2*QLL(predicted, observed)
#     loss = (weight*d)/1
    return torch.sum(d)


# def get_grad_tweedie(y_true, y_pred, roll_mat_csr, rho=1.5):
#     """
#     fixing above, workin progress
#     https://scikit-learn.org/stable/modules/model_evaluation.html#mean-poisson-gamma-and-tweedie-deviances
#     https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/metric/regression_metric.hpp#L300-L318
#     https://github.com/microsoft/LightGBM/blob/1c27a15e42f0076492fcc966b9dbcf9da6042823/src/objective/regression_objective.hpp#L702-L732
#     """
#     roll_mat_coo = roll_mat_csr.tocoo()
#     rollTensor = torch.sparse.LongTensor(torch.LongTensor([roll_mat_coo.row.tolist(), roll_mat_coo.col.tolist()]),
#                               torch.LongTensor(roll_mat_coo.data.astype(np.int32))).type(torch.FloatTensor)

#     eps = 1e-10

#     factor = 3e1
#     # factor = 3
#     if y_pred.sum() <= eps:
#         print("wtahsd!!")
#         factor = 1e20
#         # y_pred = np.random.rand(len(y_pred))

#     y_pred = np.where(y_pred<0, eps, y_pred)  #Filter 0 and negative values 

#     y_true_t = torch.FloatTensor(y_true.reshape(NUM_ITEMS, -1))
#     y_pred_t = torch.FloatTensor(y_pred.reshape(NUM_ITEMS, -1)).requires_grad_(True)

#     rolled_y_true = torch.mm(rollTensor, y_true_t).flatten() 
#     rolled_y_pred = torch.mm(rollTensor, y_pred_t).flatten() + eps

#     loss = tweedieloss(rolled_y_pred, rolled_y_true)


#     loss.backward()
#     grad = y_pred_t.grad.detach().numpy().flatten()/factor


#     debug = True
#     if debug:
#         print(f"Loss: {loss}")
#         with torch.no_grad():
#             print(f"y_true_t: {torch.mean(y_true_t)}")
#             print(f"y_pred_t: {torch.mean(y_pred_t)}")
#             print(f"min y_pred_t: {torch.min(y_pred_t)}")
#             print(f"min rolled_y_pred: {torch.min(y_pred_t)}")
#             # print(f"grad: {grad.mean()}")
#             # print(f"min grad: {grad.min()}")
#             # print(f"max grad: {grad.max()}")
#             print(grad.sum())

#     return grad  