from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from sklearn.metrics import make_scorer,mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import time
import pandas as pd
import numpy as np
import os
import json
# import matplotlib.pyplot as plt

path=os.getcwd()
parent_directory = os.path.dirname(path.strip())

target_value=pd.read_csv(os.path.join(parent_directory,'Data/csv_files/mp_fm.csv')).iloc[:,1]
X_feature=np.load("mp_features.npy")
minmax_scaler=preprocessing.MinMaxScaler()
train_value_scaler=minmax_scaler.fit_transform(X_feature)
X_train_full,X_test,y_train_full,y_test=train_test_split(train_value_scaler,target_value,
                                                         test_size=0.1,random_state=123)

def rmse(y_true, y_pred):  
    return mean_squared_error(y_true, y_pred,squared=False)

def mse(y_ture,y_pred):
    return mean_squared_error(y_ture,y_pred)

def mae(y_ture,y_pred):
    return mean_absolute_error(y_ture,y_pred)

def train_model(model, param_grid=[], X_train=[], y_train=[], X_test=[], y_test=[],
                splits=10, repeats=5):
    # Start the timer
    t1=time.time()
    
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats,random_state=42)
    # perform a grid search if param_grid given
    if len(param_grid)>0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring="neg_mean_squared_error",
                               verbose=1, return_train_score=True)
        # search the grid
        gsearch.fit(X_train,y_train)

        # extract best model from the grid
        model = gsearch.best_estimator_        
        best_idx = gsearch.best_index_
        best_parameter=gsearch.best_params_
        
        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)       
        cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
        cv_std = grid_results.loc[best_idx,'std_test_score']
    # no grid search, just cross-val score for given model    
    else:
        grid_results = []
        cv_results = cross_val_score(model, X_train,y_train, scoring="neg_mean_squared_error", cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

    # predict y using the fitted model
    start_time = time.time()
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('best parameters:',best_parameter)
    print('----------------------')
    print('train r2_score=',model.score(X_train,y_train))
    print('test r2_score=',model.score(X_test,y_test))
    print('rmse=',rmse(y_test, y_pred_test))
    print('mse=',mse(y_test, y_pred_test))
    print('mae=',mae(y_test, y_pred_test))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    print("predict time: {:.2f} seconds".format(elapsed_time))
    print("total time: {:.2f} seconds".format(end_time-t1))
#     plt.figure(figsize=(6,5))
#     plt.scatter(y_test,y_pred_test,label='test data')
#     plt.scatter(y_train,y_pred_train,label='train data')
#     plt.plot(target_value,target_value,'-k')
#     plt.legend()
#     plt.xlabel('True Value')
#     plt.ylabel('Predicted Value')
#     plt.title('test: r^2 = {:.3f}'.format(model.score(X_test,y_test)))
#     plt.show()
 
    return model, cv_score, grid_results, y_pred_train,  y_pred_test

# places to store optimal models and scores
opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])
# no. k-fold splits
splits=9
# no. k-fold iterations
repeats=5

model = 'RandomForest'
opt_models[model] = RandomForestRegressor(random_state=42)

param_grid = {'n_estimators':np.arange(50,450,50),
#               'max_depth':[2,4,6,8,10],
#               'min_samples_split':[2,4,6,8],
#              'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
#               'max_features':[1,2,3,4]
              }

opt_models[model], cv_score, grid_results, y_pred_train, y_pred_test= train_model(opt_models[model], 
                                                                                  param_grid=param_grid,
                                                        X_train=X_train_full,y_train=y_train_full,
                                                        X_test=X_test, y_test=y_test, splits=9, repeats=1)


# store model information, i.e. name, cross validation: mean and std
cv_score.name = model
score_models = score_models.append(cv_score)


model = 'GradientBoosting'
opt_models[model] = GradientBoostingRegressor(random_state=42)

param_grid = {'n_estimators':np.arange(50,450,50),
#               'learning_rate':[0.1],
#               'max_depth':[3,5,6,7],
#               'min_samples_split':[5,6,7],
#               'learning_rate':[0.02,0.05,0.1,],
             }


opt_models[model], cv_score, grid_results, y_pred_train, y_pred_test= train_model(opt_models[model], 
                                                                                  param_grid=param_grid,
                                                        X_train=X_train_full,y_train=y_train_full,
                                                        X_test=X_test, y_test=y_test, splits=9, repeats=1)



cv_score.name = model
score_models = score_models.append(cv_score)


model = 'LGBMRegressor'
opt_models[model] = LGBMRegressor(random_state=42)

'''
you can write down the hyperparameters that you want to search in the param_grid
'''
param_grid = {'n_estimators':np.arange(50,450,50),
#               'max_features':[8,12,16,20,24],
#                 "num_leaves":[31],
#               "learning_rate":[0.1]
             }

opt_models[model], cv_score, grid_results, y_pred_train, y_pred_test= train_model(opt_models[model], 
                                                                                  param_grid=param_grid,
                                                        X_train=X_train_full,y_train=y_train_full,
                                                        X_test=X_test, y_test=y_test, splits=9, repeats=1)


# store model information, i.e. name, cross validation: mean and std
cv_score.name = model
score_models = score_models.append(cv_score)


model = 'XGBRegressor'
opt_models[model] = XGBRegressor(random_state=42)

'''
you can write down the hyperparameters that you want to search in the param_grid
'''
param_grid = {'n_estimators':np.arange(50,450,50),
#               'max_features':[8,12,16,20,24],
#               "learning_rate":[0.08]
             }

opt_models[model], cv_score, grid_results, y_pred_train, y_pred_test= train_model(opt_models[model], 
                                                                                  param_grid=param_grid,
                                                        X_train=X_train_full,y_train=y_train_full,
                                                        X_test=X_test, y_test=y_test, splits=9, repeats=1)


# store model information, i.e. name, cross validation: mean and std
cv_score.name = model
score_models = score_models.append(cv_score)
