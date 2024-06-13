#Author: Max Korbmacher (max.korbmacher@gmail.com)
#
###### ADD PATH FOLDER WHERE FILES WILL BE SAVED
savepath="/Users/max/Documents/Projects/FS_brainage/version_mixing/shuffled_train_test_splits/"
###### DEFINE THE NUMBER OF REPETITIONS OF THE RANDOM SAMPLING PROCEDURE (of train-test split AND features)
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
from sklearn.linear_model import LinearRegression
import random
import joblib
#
def bic(y, y_pred, p):
    """
    Returns the BIC score of a model.

    Input:-
    y: the labelled data in shape of an array of size  = number of samples. 
        type = vector/array/list
    y_pred: predicted values of y from a regression model in shape of an array
        type = vector/array/list
    p: number of variables used for prediction in the model.
        type = int

    Output:-
    score: It outputs the BIC score
        type = int


    Tests:-
    Raise Error if length(y) <= 1 or length(y_pred) <= 1.
    Raise Error if length(y) != length(y_pred).
    Raise TypeError if y and y_pred are not vector.
    Raise TypeError if elements of y or y_pred are not integers.
    Raise TypeError if p is not an int.
    Raise Error if p < 0.
    """

    # package dependencies
    import numpy as np
    import pandas as pd
    import collections
    collections.Mapping = collections.abc.Mapping
    collections.Sequence = collections.abc.Sequence

    # Input type error exceptions
    if not isinstance(y, (collections.Sequence, np.ndarray, pd.core.series.Series)):
        raise TypeError("Argument 1 not like an array.")

    if not isinstance(y_pred, (collections.Sequence, np.ndarray, pd.core.series.Series)):
        raise TypeError("Argument 2 not like an array.")

    for i in y:
        if not isinstance(i, (int, float)):
            raise TypeError("All elements of argument 1 must be int or float.")

    for i in y_pred:
        if not isinstance(i, (int, float)):
            raise TypeError("All elements of argument 2 must be int or float.")

    if not isinstance(p, (int, float)):
        raise TypeError("'Number of variables' must be of type int or float.")

    if p <= 0:
        raise TypeError("'Number of variables' must be positive integer.")

    if isinstance(p, int) != True:
        raise TypeError("Expect positive integer")

    if len(y) <= 1 or len(y_pred) <= 1:
        raise TypeError("observed and predicted values must be greater than 1")

    # Length exception
    if not len(y) == len(y_pred):
        raise TypeError("Equal length of observed and predicted values expected.")
    else:
        n = len(y)

    # Score

    residual = np.subtract(y_pred, y)
    SSE = np.sum(np.power(residual, 2))
    BIC = n*np.log(SSE/n) + p*np.log(n)
    return BIC
def aic(y, y_pred, p):
    """
    Return an AIC score for a model.

    Input:
    y: array-like of shape = (n_samples) including values of observed y
    y_pred: vector including values of predicted y
    p: int number of predictive variable(s) used in the model

    Output:
    aic_score: int or float AIC score of the model

    Raise TypeError if y or y_pred are not list/tuple/dataframe column/array.
    Raise TypeError if elements in y or y_pred are not integer or float.
    Raise TypeError if p is not int.
    Raise InputError if y or y_pred are not in same length.
    Raise InputError if length(y) <= 1 or length(y_pred) <= 1.
    Raise InputError if p < 0.
    """

    # Package dependencies
    import numpy as np
    import pandas as pd

    # User-defined exceptions
    class InputError(Exception):
        """
        Raised when there is any error from inputs that no base Python exceptions cover.
        """
        pass

    # Check conditions:
    ## Type condition 1: y and y_pred should be array-like containing numbers
    ### check type of y and y_pred
    if isinstance(y, (np.ndarray, list, tuple, pd.core.series.Series)) == False or isinstance(y_pred, (np.ndarray, list, tuple, pd.core.series.Series)) == False:
        raise TypeError("Expect array-like shape (e.g. array, list, tuple, data column)")
    ### check if elements of y and y_pred are numeric
    else:
        for i in y:
            for j in y_pred:
                if isinstance(i, (int, float)) != True or isinstance(j, (int, float)) != True:
                    raise TypeError("Expect numeric elements in y and y_pred")

    ## Type condition 2: p should be positive integer
    ### check if p is integer
    if isinstance(p, int) != True:
        raise TypeError("Expect positive integer")
    ### check if p is positive
    elif p <= 0:
        raise InputError("Expect positive integer")

    ## Length condition: length of y and y_pred should be equal, and should be more than 1
    ### check if y and y_pred have equal length
    if not len(y) == len(y_pred):
        raise InputError("Expect equal length of y and y_pred")
    ### check if y and y_pred length is larger than 1
    elif len(y) <= 1 or len(y_pred) <= 1:
        raise InputError("Expect length of y and y_pred to be larger than 1")
    else:
        n = len(y)

    # Calculation
    resid = np.subtract(y_pred, y)
    rss = np.sum(np.power(resid, 2))
    aic_score = n*np.log(rss/n) + 2*p

    return aic_score
#
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
#
from timeit import default_timer as timer
from datetime import timedelta
#Loading the saved pretrained models on single versions
#(these serve as comparison and predict in the randomly split data in addition to the version-mixed models.)
with open('/Users/max/Documents/Projects/FS_brainage/FS5/results/LM_model.pkl', 'rb') as pickle_file:
    FS5_model = pickle.load(pickle_file)
with open('/Users/max/Documents/Projects/FS_brainage/FS7/results/LM_model.pkl', 'rb') as pickle_file:
    FS7_model = pickle.load(pickle_file)
#
print("Start timer.")
start = timer()
#
####################################
print("We start with sampling from FS5 and FS7 data to create new version mixed data frames")
# load data
FS5_train = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS5_test = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS7_train = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")
FS7_test = pd.read_csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
# sort data by eid
FS5_train = FS5_train.sort_values(by=['eid'])
FS5_test = FS5_test.sort_values(by=['eid'])
FS7_train = FS7_train.sort_values(by=['eid'])
FS7_test = FS7_test.sort_values(by=['eid'])
# merge data to be then split at a later time point
FS5 = pd.concat([FS5_train, FS5_test], axis = 0)
FS7 = pd.concat([FS7_train, FS7_test], axis = 0)
# set seed for reproduciblity of analyses
random.seed(1234)
#
# Create a loop for the sampling procedure.
# We can call the frames "model", as they produce different models
# The frame which contains the "models" is called frames (simply said the 1000x1000 data frames)
frame = {}
model = {} # effectively training data
test_data = {}
for i in range(number):
    random.seed(1234)
    ######## STEP 1: TRAIN-TEST SPLIT
    # sample 50% of the FS5 data to establish the FS5 training data
    FS5_train = FS5.sample(frac=0.5)
    # restrict the FS7 training data to the same selection of individuals
    FS7_train = FS7.loc[FS7['eid'].isin(FS5_train['eid'].values.astype(list))]
    # the IDs which are not contained in the training data are the test data, hah!
    FS7_test = FS7.loc[~FS7['eid'].isin(FS5_train['eid'].values.astype(list))]
    FS5_test = FS5.loc[~FS5['eid'].isin(FS5_train['eid'].values.astype(list))]
    #
    ######## STEP 2: FS5-FS7 SPLIT
    FS5_train_portion = FS5_train.sample(frac=0.5)
    # the other half will be FS7 training data
    FS7_train_portion = FS7_train.loc[~FS7_train['eid'].isin(FS5_train_portion['eid'].values.astype(list))]
    # concatinate the two frames
    dat = (pd.concat([FS5_train_portion, FS7_train_portion]))
    dat = dat.sort_values(by=['eid'])
    model[i] = dat.copy()
    # we also need to sample randomly from the test data
    # sample 50% of the FS5 testing data randomly
    FS5_test_portion = FS5_test.sample(frac=0.5)
    # the other half will be FS7 testing data
    FS7_test_portion = FS7_test.loc[~FS7_test['eid'].isin(FS5_test_portion['eid'].values.astype(list))]
    # concatinate the two frames
    dat1 = (pd.concat([FS5_test_portion, FS7_test_portion]))
    dat1 = dat1.sort_values(by=['eid'])
    test_data[i] = dat1.copy()
# Now, prep also the FS7 and FS5 data frames for later predictions
FS5age = FS5_test['Age']
FS7age = FS7_test['Age']
FS7eid = FS7_test['eid']
FS5eid = FS7_test['eid']
FS5_test = FS5_test.drop('eid',axis = 1)
FS5_test = FS5_test.drop('Age',axis = 1)
FS5_test = FS5_test.drop('Sex',axis = 1)
FS5_test = FS5_test.drop('Scanner',axis = 1)
FS7_test = FS7_test.drop('eid',axis = 1)
FS7_test = FS7_test.drop('Age',axis = 1)
FS7_test = FS7_test.drop('Sex',axis = 1)
FS7_test = FS7_test.drop('Scanner',axis = 1)
print(number, "randomly sampled data frames created. Training starts for each of these.")
#
####################################
#
# create some additional dictionaries for looping procedures (for data, correlation coefficients, and training & test preds)
dict = {}
corr = {}
pred = {}
pred_test = {}
FS52mix = {}
FS72mix = {}
mix2FS5 = {}
mix2FS7 = {}
# we also need dicts for output eval metrics
#eval_metrics_train = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#eval_metrics_test =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#eval_metrics_test_FS52mix =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#eval_metrics_test_FS72mix =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#eval_metrics_test_mix2FS5 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#eval_metrics_test_mix2FS7 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#
test_eval = []
train_eval = []
train_AIC = []
train_BIC = []
train_MAE = []
train_RMSE = []
train_R2 = []
train_r = []
test_AIC = []
test_BIC = []
test_MAE = []
test_RMSE = []
test_R2 = []
test_r = []
#
#
# And an output dictionary to store r and CI in json (to use for LaTeX table in step 8)
out_dict = {}
# Finally a dictionary which will contain the models
mods = {}
#
####################################
#
# start training
for m in model:
    # #saves the results to text files
    # file=savepath+'Output_LM_%s.txt'%m
    # with open(file, 'w') as text_file:
    # text_file.write("===============\n")
    # text_file.write("MODEL = %s\n" %m)
    # text_file.write("===============\n")
    dict['%s' % m] = model[m]
    #To test run with smaller N:
    #dict['%s' % m] = dict['%s' % m].sample(frac=0.1)
    # CHECK FILE CONTENT AND LENGTH
    print ('printing head of for iteration number %s' %m)
    print (dict['%s' % m].head(5))
    print ('printing number of columns for iteration number %s' %m)
    print (len(dict['%s' % m].columns))
    print ('printing length of datafile for iteration number %s' %m)
    print (len(dict['%s' % m]))
    # SPLIT THE FILE INTO X AND Y, WHERE X IS ALL THE MRI DATA AND Y IS AGE
    x = dict['%s' % m]
    print ('splitting data into x and y for %s' % m)
    y = x['Age']
    # MAKE A COPY OF THE DATA FRAME TO MERGE WITH ESTIMATED BRAIN AGE AT THE END OF SCRIPT
    x_copy = x.copy()
    # REMOVE VARIABLES FROM X THAT SHOULD NOT BE INCLUDED IN THE REGRESSOR
    x = x.drop('eid',axis = 1)
    x = x.drop('Age',axis = 1)
    x = x.drop('Sex',axis = 1)
    x = x.drop('Scanner',axis = 1)
    # CHECK THAT X INCLUDES ONLY MRI VARIABLES, AND Y INCLUDES ONLY AGE
    print ('printing final x for iteration number %s' % m)
    print (x.head(5))
    print ('printing final y for iteration number %s' % m)
    print (y.head(5))
    # SPECIFY MODEL
    # # configure cross-validation procedure
    # cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # # define the model
    # the presented hyperparameters were defined based on previous multimodal brain age predictions.
    search = LinearRegression()
    # ADD PREDICTED BRAIN AGE TO X_COPY TO GET A FULL DATAFRAME WITH ALL VARIABLES
    # fit the model
    result = search.fit(x,y)
    ## predictions in training data
    pred[m] = result.predict(x)
    x_copy['pred_age_training_%s' % m] = pred[m]
    #
    # Save the model on the disk
    #with open(savepath+ 'LM_model_%s.pkl' % m,'wb') as f:
    #    pickle.dump(result,f)
    # make predictions
    print("Training completed. Predictions start for iteration %s." %m)
    #
    ## predictions in test data
    y_copy = test_data[m].copy()
    ### remove unsee variables (demographics)
    test_data[m] = test_data[m].drop('eid',axis = 1)
    test_data[m] = test_data[m].drop('Age',axis = 1)
    test_data[m] = test_data[m].drop('Sex',axis = 1)
    test_data[m] = test_data[m].drop('Scanner',axis = 1)
    ### predict
    pred_test[m] = result.predict(test_data[m])
    y_copy['pred_age_test_%s' % m] = pred_test[m]
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
    #x_copy_save.to_csv(savepath+'Brainage_LM_Training_%s.csv' %m, sep=',',index=None)
    #y_copy_save.to_csv(savepath+'Brainage_LM_Testing_%s.csv' %m, sep=',',index=None)
    #
    # EVALUATE PREDICTIONS IN TEST AND TRAINING DATA
    #
    ##### both randomly drawn
    #
    p = len(x.columns) # number of features
    train_AIC.append(aic(x_copy['Age'], x_copy['pred_age_training_%s' %m], p))
    train_BIC.append(bic(x_copy['Age'], x_copy['pred_age_training_%s' %m], p))
    train_MAE.append(mean_absolute_error(x_copy['Age'], x_copy['pred_age_training_%s' %m]))
    train_RMSE.append(root_mean_squared_error(x_copy['Age'], x_copy['pred_age_training_%s' %m]))
    train_R2.append(r2_score(x_copy['Age'], x_copy['pred_age_training_%s' %m]))
    train_r.append(np.corrcoef(x_copy['Age'], x_copy['pred_age_training_%s' %m])[1,0])
    test_AIC.append(aic(y_copy['Age'], y_copy['pred_age_test_%s' %m], p))
    test_BIC.append(bic(y_copy['Age'], y_copy['pred_age_test_%s' %m], p))
    test_MAE.append(mean_absolute_error(y_copy['Age'], y_copy['pred_age_test_%s' %m]))
    test_RMSE.append(root_mean_squared_error(y_copy['Age'], y_copy['pred_age_test_%s' %m]))
    test_R2.append(r2_score(y_copy['Age'], y_copy['pred_age_test_%s' %m]))
    test_r.append(np.corrcoef(y_copy['Age'], y_copy['pred_age_test_%s' %m])[1,0])
    #
    #
    ###### from FS5 to randomly drawn (cannot be done in current setup due to train-test overlap)
    #FS52mix[m] = FS5_model.predict(test_data[m])
    ###### from FS7 to randomly drawn (cannot be done in current setup due to train-test overlap)
    #FS72mix[m] = FS7_model.predict(test_data[m])
    ###### from randomly drawn to FS5 (cannot be done in current setup due to train-test overlap)
    #mix2FS5[m] = result.predict(FS5_test) 
    ###### from randomly drawn to FS7 (cannot be done in current setup due to train-test overlap)
    #mix2FS7[m] = result.predict(FS7_test)
    # add predictions into single df for sorting
    #y_copy['pred_age_FS52mix%s' % m] = FS52mix[m]
    #y_copy['pred_age_FS72mix%s' % m] = FS72mix[m]
    #y_copy['pred_age_mix2FS5%s' % m] = mix2FS5[m]
    #y_copy['pred_age_mix2FS7%s' % m] = mix2FS7[m]
    #
    # order / sort predictions
    #y_copy = y_copy.sort_values(by=['eid'])
    #
    # then create data frames for predictions
    #pred_df.append(x_copy['pred_age_training_%s' %m]) # Mix2Mix training data
    #pred_test_df.append(y_copy['pred_age_test_%s' %m]) # Mix2Mix test data
#
#
# FIRST: BAGGING APPROACH
# average across the i = number predictions in order to compare them statistically
#pred_df1 = pd.DataFrame.from_dict(pred_df) # Mix2Mix training data
#pred_test_df1 = pd.DataFrame.from_dict(pred_test_df) # Mix2Mix test data
# mix2FS5_df1 = pd.DataFrame.from_dict(mix2FS5_df)
# mix2FS7_df1 = pd.DataFrame.from_dict(mix2FS7_df)
# FS52mix_df1 = pd.DataFrame.from_dict(FS52mix_df)
# FS72mix_df1 = pd.DataFrame.from_dict(FS72mix_df)
# transpose frames for having the correct format (with each iteration being a column)
#pred_df1 = pred_df1.transpose()
#pred_test_df1 = pred_test_df1.transpose()
# mix2FS5_df1 = mix2FS5_df1.transpose()
# mix2FS7_df1 = mix2FS7_df1.transpose()
# FS52mix_df1 = FS52mix_df1.transpose()
# FS72mix_df1 = FS72mix_df1.transpose()
# average across data frames
# mean_pred = pred_df1.mean(axis=1)
# mean_pred_test  = pred_test_df1.mean(axis=1)
# mean_mix2FS5 = mix2FS5_df1.mean(axis=1)
# mean_mix2FS7 = mix2FS7_df1.mean(axis=1)
# mean_FS52mix = FS52mix_df1.mean(axis=1)
# mean_FS72mix = FS72mix_df1.mean(axis=1)
#
# Then put the resulting preds and age into one new df.
# Note that we do NOT consider training predictions here, we only compare test performance!
#predictions = pd.DataFrame({'eid':(FS7eid), 'age':(FS7age), 'mix_test':(mean_pred_test),'mix2FS5':(mean_mix2FS5),'mix2FS7':(mean_mix2FS7),'FS52mix':(mean_FS52mix),'FS72mix':(mean_FS72mix)})
# The predictions are then saved as data frames
#predictions.to_csv(savepath+'mean_predictions.csv', sep=',', index=False)
#
#
# NEXT: LOOK AT EVALUATION METRICS
#
# we combine the dictionaries into data frames
eval_metrics_train = pd.concat([pd.DataFrame.from_dict(train_R2), pd.DataFrame.from_dict(train_r), pd.DataFrame.from_dict(train_MAE), pd.DataFrame.from_dict(train_RMSE), pd.DataFrame.from_dict(train_AIC), pd.DataFrame.from_dict(train_BIC)], axis = 1)
eval_metrics_test =  pd.concat([pd.DataFrame.from_dict(test_R2), pd.DataFrame.from_dict(test_r), pd.DataFrame.from_dict(test_MAE), pd.DataFrame.from_dict(test_RMSE), pd.DataFrame.from_dict(test_AIC), pd.DataFrame.from_dict(test_BIC)], axis = 1)
#
# and label the metrics according to the model
eval_metrics_train.columns = ['R2_mix_train', 'r_mix_train', 'MAE_mix_train', 'RMSE_mix_train', 'AIC', 'BIC']
eval_metrics_test.columns = ['R2_mix_test', 'r_mix_test', 'MAE_mix_test', 'RMSE_mix_test', 'AIC', 'BIC']
# 
print("Mean and Standard Deviation for training and test data prediction evaluation metrics.")
#singleframe = pd.concat([eval_metrics_train,eval_metrics_test, eval_metrics_test_FS52mix, eval_metrics_test_FS72mix, eval_metrics_test_mix2FS5, eval_metrics_test_mix2FS7],axis = 1)
singleframe = pd.concat([eval_metrics_train,eval_metrics_test],axis = 1)
sumstats = {"Mean": singleframe.mean(), "SD": singleframe.std()}
sumstats = pd.DataFrame(sumstats)
# save the performance frame
sumstats.to_csv(savepath+'shuffled_training_and_testing_sumstats_LM.csv', sep=',')
# done
print("Elapsed time:")
end = timer()
print(timedelta(seconds=end-start))
print("########################")
print("Interpretation time!")
