# Validate findings on brain age predictions from different FreeSurfer versions.
# This is done by using repeated random sampling of training and test data.
#
# Author: Max Korbmacher (max.korbmacher@gmail.com)
# 12 April 2024
#
###### ADD PATH FOLDER WHERE FILES WILL BE SAVED
savepath="/Users/max/Documents/Projects/FS_brainage/results/repeats/"
###### DEFINE THE NUMBER OF REPETITIONS OF THE RANDOM SAMPLING PROCEDURE
number = 1000
###########################
#
# import packages
#
import csv
import pandas as pd
from functools import reduce
import numpy as np
#import pingouin as pg
#from pingouin import partial_corr
#from pingouin import logistic_regression
import scipy
from scipy.stats.stats import pearsonr
import time
from numpy import mean
from numpy import std
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.feature_selection import r_regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
from scipy import stats
import sys, os
import statsmodels.api as sm
import json
from sklearn.linear_model import Lasso
import random
import joblib
#
def pearsonr_ci(x,y,alpha=0.05):

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi
#
import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=2)
sns.set_style("white")
#
#Using this to get LaTeX font for plots (LaTeX code rules must be followed for e.g. axis titles (no underscores etc without \))
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
from timeit import default_timer as timer
from datetime import timedelta
print("Timer starts now.")
start = timer()
#
####################################
print("We start with sampling from FS5 and FS7 data to create new train-test splits, keeping the ratio constant at 50% both.")
# load data
FS5_train = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS5_test = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS7_train = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")
FS7_test = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
# merge data to be then split at a later time point
FS5 = pd.concat([FS5_train, FS5_test], axis = 0)
FS7 = pd.concat([FS7_train, FS7_test], axis = 0)
#
# set seed for reproduciblity of analyses
random.seed(1234)
#
# Create a loop for the sampling procedure producing equal train-test splits for different versions
FS5_train = {}
FS5_test = {}
FS7_train = {}
FS7_test = {}
for i in range(number):
    random.seed(1234)
    # sample 50% of the FS5 data to establish the FS5 training data
    FS5_train[i] = FS5.sample(frac=0.5)
    # restrict the FS7 training data to the same selection of individuals
    FS7_train[i] = FS7.loc[FS7['eid'].isin(FS5_train[i]['eid'].values.astype(list))]
    # the IDs which are not contained in the training data are the test data, hah!
    FS7_test[i] = FS7.loc[~FS7['eid'].isin(FS5_train[i]['eid'].values.astype(list))]
    FS5_test[i] = FS5.loc[~FS5['eid'].isin(FS5_train[i]['eid'].values.astype(list))]
print(number, "randomly sampled data frames created. Training starts for each of these.")
#
####################################
#
# create some additional dictionaries for outputs
dict = {}
dict2 = {}
dict3 = {}
corr = {}
pred = {}
pred_FS52FS5 = {}
pred_FS52FS7 = {}
pred_FS72FS7 = {}
pred_FS72FS5 = {}
# we also need dicts for output eval metrics
eval_metrics_train_FS52FS5 = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_train_FS72FS7 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_FS52FS5 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_FS52FS7 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_FS72FS7 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_FS72FS5 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#
FS52FS5_train_eval = []
FS72FS7_train_eval = []
FS52FS5_eval = []
FS52FS7_eval = []
FS72FS7_eval = []
FS72FS5_eval = []
# And an output dictionary to store r and CI in json (to use for LaTeX table in step 8)
out_dict = {}
# Finally a dictionary which will contain the models
mods = {}
#
####################################
#
# start training
for m in FS5_train:
    # define the training data
    dict['%s' % m] = FS5_train[m]
    dict2['%s' % m] = FS7_train[m]
    dict3['%s' % m] = FS7_test[m]
    #To test run with smaller N:
    #dict['%s' % m] = dict['%s' % m].sample(frac=0.1)
    #dict2['%s' % m] = dict2['%s' % m].sample(frac=0.1)
    # SPLIT THE FILE INTO X AND Y, WHERE X IS ALL THE MRI DATA AND Y IS AGE
    x = dict['%s' % m]
    x2 = dict2['%s' % m]
    x3 = dict3['%s' % m]
    print ('splitting data into x and y for iteration %s' % m)
    y = x['Age']
    y2 = x2['Age']
    # MAKE A COPY OF THE DATA FRAMES FOR EASIER PROCESSING LATER ON
    x_copy = x.copy()
    x2_copy = x2.copy()
    y_copy = x3.copy()
    # REMOVE VARIABLES FROM X THAT SHOULD NOT BE INCLUDED IN THE TRAINING PROCEDURE
    x = x.drop('eid',axis = 1)
    x = x.drop('Age',axis = 1)
    x = x.drop('Sex',axis = 1)
    x = x.drop('Scanner',axis = 1)
    x2 = x2.drop('eid',axis = 1)
    x2 = x2.drop('Age',axis = 1)
    x2 = x2.drop('Sex',axis = 1)
    x2 = x2.drop('Scanner',axis = 1)
    # CHECK THAT X INCLUDES ONLY MRI VARIABLES, AND Y INCLUDES ONLY AGE
    print ('printing final x for iteration number %s' % m)
    print (x.head(5))
    print ('printing final y for iteration number %s' % m)
    print (y.head(5))
    # define the model
    # the presented hyperparameters are default parameters with a higher # of max iterations
    our_model = Lasso(alpha=1.0, max_iter=100000)
    # fit the model for predictions in test data
    result = our_model.fit(x,y)
    result2 = our_model.fit(x2,y2)
    # Save the model on the disk (if this is wished)
    #with open(savepath+ 'Lasso_model_%s.pkl' % m,'wb') as f:
    #    pickle.dump(result,f)
    # make predictions
    print("Training completed. Predictions start for iteration %s." %m)
    #
    ## predictions in training data
    train_BA_FS5 = result.predict(x)
    train_BA_FS7 = result2.predict(x2)
    ## predictions in test data
    FS5_test[m] = FS5_test[m].drop('eid',axis = 1)
    FS5_test[m] = FS5_test[m].drop('Age',axis = 1)
    FS5_test[m] = FS5_test[m].drop('Sex',axis = 1)
    FS5_test[m] = FS5_test[m].drop('Scanner',axis = 1)
    FS7_test[m] = FS7_test[m].drop('eid',axis = 1)
    FS7_test[m] = FS7_test[m].drop('Age',axis = 1)
    FS7_test[m] = FS7_test[m].drop('Sex',axis = 1)
    FS7_test[m] = FS7_test[m].drop('Scanner',axis = 1)
    ### predict
    pred_FS52FS5[m]= result.predict(FS5_test[m])
    pred_FS52FS7[m]= result.predict(FS7_test[m])
    pred_FS72FS7[m]= result2.predict(FS7_test[m])
    pred_FS72FS5[m]= result2.predict(FS5_test[m])
    #
    #
    # SAVE FILE WITH PREDICTIONS
    # First create a dataframe including only relevant variables
    #x_copy_save = x_copy[['eid','Age','pred_age_training_%s' % m]]
    #y_copy_save = y_copy[['eid','Age','pred_age_test_%s' % m]]
    #
    #
    #print (x_copy_save.head(5))	#uncomment to check file content
    #print ('saving file with brain age estimates')
    #x_copy_save.to_csv(savepath+'Brainage_Lasso_Training_%s.csv' %m, sep=',',index=None)
    #x_copy_save.to_csv(savepath+'Brainage_Lasso_Testing_%s.csv' %m, sep=',',index=None)
    #
    # EVALUATE PREDICTIONS IN TEST AND TRAINING DATA
    #
    ##### training data: calculate model evalutation metrics
    #
    ev1 =  {mean_absolute_error(x_copy['Age'], train_BA_FS5),root_mean_squared_error(x_copy['Age'], train_BA_FS5),r2_score(x_copy['Age'], train_BA_FS5),np.corrcoef(x_copy['Age'], train_BA_FS5)[1,0]}
    ev2 =  {mean_absolute_error(x2_copy['Age'], train_BA_FS7),root_mean_squared_error(x2_copy['Age'], train_BA_FS7),r2_score(x2_copy['Age'], train_BA_FS7),np.corrcoef(x2_copy['Age'], train_BA_FS7)[1,0]}
    # create vectors with respective metrics
    FS52FS5_train_eval.append(ev1)
    FS72FS7_train_eval.append(ev2)
    #
    ##### test data: calculate model evalutation metrics
    #         
    ev3 =  {mean_absolute_error(y_copy['Age'], pred_FS52FS5[m]),root_mean_squared_error(y_copy['Age'], pred_FS52FS5[m]),r2_score(y_copy['Age'], pred_FS52FS5[m]),np.corrcoef(y_copy['Age'], pred_FS52FS5[m])[1,0]}
    ev4 =  {mean_absolute_error(y_copy['Age'], pred_FS52FS7[m]),root_mean_squared_error(y_copy['Age'], pred_FS52FS7[m]),r2_score(y_copy['Age'], pred_FS52FS7[m]),np.corrcoef(y_copy['Age'], pred_FS52FS7[m])[1,0]}
    ev5 =  {mean_absolute_error(y_copy['Age'], pred_FS72FS7[m]),root_mean_squared_error(y_copy['Age'], pred_FS72FS7[m]),r2_score(y_copy['Age'], pred_FS72FS7[m]),np.corrcoef(y_copy['Age'], pred_FS72FS7[m])[1,0]}
    ev6 =  {mean_absolute_error(y_copy['Age'], pred_FS72FS5[m]),root_mean_squared_error(y_copy['Age'], pred_FS72FS5[m]),r2_score(y_copy['Age'], pred_FS72FS5[m]),np.corrcoef(y_copy['Age'], pred_FS72FS5[m])[1,0]}
    # create vectors with respective metrics
    FS52FS5_eval.append(ev3)
    FS52FS7_eval.append(ev4)
    FS72FS7_eval.append(ev5)
    FS72FS5_eval.append(ev6)
    ###########################################################################
# we combine the dictionaries into a data frame
eval_metrics_train1 = pd.DataFrame.from_dict(FS52FS5_train_eval)
eval_metrics_train2 = pd.DataFrame.from_dict(FS72FS7_train_eval)
eval_metrics_test1 =  pd.DataFrame.from_dict(FS52FS5_eval)
eval_metrics_test2 =  pd.DataFrame.from_dict(FS52FS7_eval)
eval_metrics_test3 =  pd.DataFrame.from_dict(FS72FS7_eval)
eval_metrics_test4 =  pd.DataFrame.from_dict(FS72FS5_eval)
eval_metrics_train1.columns = ['R2_FS5_train', 'r_FS5_train', 'MAE_FS5_train', 'RMSE_FS5_train']
eval_metrics_train2.columns = ['R2_FS7_train', 'r_FS7_train', 'MAE_FS7_train', 'RMSE_FS7_train']
eval_metrics_test1.columns = ['R2_FS52FS5', 'r_FS52FS5', 'MAE_FS52FS5', 'RMSE_FS52FS5']
eval_metrics_test2.columns = ['R2_FS52FS7', 'r_FS52FS7', 'MAE_FS52FS7', 'RMSE_FS52FS7']
eval_metrics_test3.columns = ['R2_FS72FS7', 'r_FS72FS7', 'MAE_FS72FS7', 'RMSE_FS72FS7']
eval_metrics_test4.columns = ['R2_FS72FS5', 'r_FS72FS5', 'MAE_FS72FS5', 'RMSE_FS72FS5']
# finally, we can look at the prediction performance across models
print("Mean and Standard Deviation for training and test data prediction evaluation metrics.")
singleframe = pd.concat([eval_metrics_train1,eval_metrics_train2,eval_metrics_test1,eval_metrics_test2,eval_metrics_test3,eval_metrics_test4],axis = 1)
sumstats = {"Mean": singleframe.mean(), "SD": singleframe.std()}
sumstats = pd.DataFrame(sumstats)
print(sumstats)
# save the performance frame
sumstats.to_csv(savepath+'random_train_test_sumstats.csv', sep=',')
# done
print("Elapsed time:")
end = timer()
print(timedelta(seconds=end-start))
print("########################")
print("Interpretation time!")
