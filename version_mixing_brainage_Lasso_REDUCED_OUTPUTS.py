#Author: Max Korbmacher (max.korbmacher@gmail.com)
#
###### ADD PATH FOLDER WHERE FILES WILL BE SAVED
savepath="/Users/max/Documents/Projects/FS_brainage/version_mixing/"
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
#
from timeit import default_timer as timer
from datetime import timedelta
#Loading the saved pretrained models on single versions
with open('/Users/max/Documents/Projects/FS_brainage/FS5/results/Lasso_model.pkl', 'rb') as pickle_file:
    FS5_model = pickle.load(pickle_file)
with open('/Users/max/Documents/Projects/FS_brainage/FS7/results/Lasso_model.pkl', 'rb') as pickle_file:
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
# set seed for reproduciblity of analyses
random.seed(1234)
#
# Create a loop for the sampling procedure.
# We can call the frames "model", as they produce different models
model = {}
for i in range(number):
    random.seed(1234)
    # sample 50% of the FS5 training data randomly
    FS5_train_portion = FS5_train.sample(frac=0.5)
    # the other half will be FS7 training data
    FS7_train_portion = FS7_train.loc[~FS7_train['eid'].isin(FS5_train_portion['eid'].values.astype(list))]
    # concatinate the two frames
    dat = (pd.concat([FS5_train_portion, FS7_train_portion]))
    model[i] = dat.copy()
# we also need to sample randomly from the test data
test_data = {}
for i in range(number):
    random.seed(1234)
    # sample 50% of the FS5 testing data randomly
    FS5_test_portion = FS5_test.sample(frac=0.5)
    # the other half will be FS7 testing data
    FS7_test_portion = FS7_test.loc[~FS7_test['eid'].isin(FS5_test_portion['eid'].values.astype(list))]
    # concatinate the two frames
    dat1 = (pd.concat([FS5_test_portion, FS7_test_portion]))
    test_data[i] = dat1.copy()
# Now, prep also the FS7 and FS5 data frames for later predictions
FS5age = FS5_test['Age']
FS7age = FS7_test['Age']
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
# we also need dicts for output eval metrics
eval_metrics_train = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_FS52mix =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_FS72mix =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_mix2FS5 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
eval_metrics_test_mix2FS7 =  pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'r'])
#
test_eval = []
train_eval = []
FS52mix_eval = []
FS72mix_eval = []
mix2FS5_eval = []
mix2FS7_eval = []
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
    # file=savepath+'Output_Lasso_%s.txt'%m
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
    search = Lasso(max_iter=100000)
        # text_file.write ('validating %s model\n' % m)

        # text_file.write ('------------------------------\n')
        # text_file.write ('RMSE values:\n')
        # RMSE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_root_mean_squared_error',n_jobs = 4)
        # text_file.write('Mean and STD for RMSE: %.3f (%.3f)\n' % (mean(RMSE), std(RMSE)))

        # text_file.write ('------------------------------\n')
        # text_file.write ('MAE values:\n')
        # MAE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_mean_absolute_error',n_jobs = 4)
        # text_file.write('Mean and STD for MAE: %.3f (%.3f)\n' % (mean(MAE), std(MAE)))

        # text_file.write ('------------------------------\n')
        # text_file.write ('R2 values:\n')
        # R2 = cross_val_score(search, x, y, cv=cv_outer,scoring='r2',n_jobs = 4)
        # text_file.write('Mean and STD for R2: %.3f (%.3f)\n' % (mean(R2), std(R2)))
            


        # # RUN CROSS_VAL_PREDICT
        # print ('Running cross_val_predict')

        # # DEFINE THE VARIABLES PRED (PREDICTED AGE) AND BAG (BRAIN AGE GAP)
        # pred[m] = cross_val_predict(search, x, y, cv=cv_outer, n_jobs=2)
        # BAG = pred[m] - y
        # #
        #

    # ADD PREDICTED BRAIN AGE TO X_COPY TO GET A FULL DATAFRAME WITH ALL VARIABLES
    # fit the model
    result = search.fit(x,y)
    ## predictions in training data
    pred[m] = result.predict(x)
    x_copy['pred_age_training_%s' % m] = pred[m]
    #
    # Save the model on the disk
    #with open(savepath+ 'Lasso_model_%s.pkl' % m,'wb') as f:
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
    #x_copy_save.to_csv(savepath+'Brainage_Lasso_Training_%s.csv' %m, sep=',',index=None)
    #y_copy_save.to_csv(savepath+'Brainage_Lasso_Testing_%s.csv' %m, sep=',',index=None)
    #
    # EVALUATE PREDICTIONS IN TEST AND TRAINING DATA
    #
    ##### both randomly drawn
    #
    ev1 =  {mean_absolute_error(x_copy['Age'], x_copy['pred_age_training_%s' %m]),root_mean_squared_error(x_copy['Age'], x_copy['pred_age_training_%s' %m]),r2_score(x_copy['Age'], x_copy['pred_age_training_%s' %m]),np.corrcoef(x_copy['Age'], x_copy['pred_age_training_%s' %m])[1,0]}
    ev2 =  {mean_absolute_error(y_copy['Age'], y_copy['pred_age_test_%s' %m]),root_mean_squared_error(y_copy['Age'], y_copy['pred_age_test_%s' %m]),r2_score(y_copy['Age'], y_copy['pred_age_test_%s' %m]),np.corrcoef(y_copy['Age'], y_copy['pred_age_test_%s' %m])[1,0]}
    train_eval.append(ev1)
    test_eval.append(ev2)
    #
    #
    ###### from FS5 to randomly drawn
    #
    #
    FS52mix = FS5_model.predict(test_data[m])
    #
    #
    ###### from FS7 to randomly drawn
    FS72mix = FS7_model.predict(test_data[m])
    #
    #
    ###### from randomly drawn to FS5
    mix2FS5 = result.predict(FS5_test) 
    ###### from randomly drawn to FS7
    mix2FS7 = result.predict(FS7_test)
    #
    # calculate model evalutation metrics
    ev3 =  {mean_absolute_error(y_copy['Age'], FS52mix),root_mean_squared_error(y_copy['Age'], FS52mix),r2_score(y_copy['Age'], FS52mix),np.corrcoef(y_copy['Age'], FS52mix)[1,0]}
    ev4 =  {mean_absolute_error(y_copy['Age'], FS72mix),root_mean_squared_error(y_copy['Age'], FS72mix),r2_score(y_copy['Age'], FS72mix),np.corrcoef(y_copy['Age'], FS72mix)[1,0]}
    ev5 =  {mean_absolute_error(FS5age, mix2FS5),root_mean_squared_error(FS5age, mix2FS5),r2_score(FS5age, mix2FS5),np.corrcoef(FS5age, mix2FS5)[1,0]}
    ev6 =  {mean_absolute_error(FS7age, mix2FS7),root_mean_squared_error(FS7age, mix2FS7),r2_score(FS7age, mix2FS7),np.corrcoef(FS7age, mix2FS7)[1,0]}
    # create vectors with respective metrics
    FS52mix_eval.append(ev3)
    FS72mix_eval.append(ev4)
    mix2FS5_eval.append(ev5)
    mix2FS7_eval.append(ev6)
        #
        #
        #
        #
        #
        #Store r and CI in dict for writing to json
        #out_dict[m] = corr[m]
        #
        #
        #Write the JSON with r and CI values to json
        #with open(savepath+'r_vals_%s.json' %m,'w') as f:
        #    json.dump(out_dict, f, sort_keys=True, indent=4)
        #
        ###########################################################################
        # WE DO NOT ESTIMATE PERMUTATION FEATURE IMPORTANCE HERE DUE TO COMP COSTS!
        ###########################################################################
        # estimate permutation feature importance
        #keys = list(feature_importances.keys())
        #values = list(feature_importances.values())
        #data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        #data.to_csv(savepath+'SVR_Feature_Importances_%s.csv' %m, sep=',')
        #print('Estimate permutation feature importance.')
        #r = permutation_importance(result, x, y,
        #                           n_repeats=10,
        #                           random_state=0,
        #                           n_jobs = 4)
        #for i in r.importances_mean.argsort()[::-1]:
        #    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        #        text_file.write(f"{x.columns[i]:<8} "
        #              f"{r.importances_mean[i]:.6f}"
        #              f" +/- {r.importances_std[i]:.6f}"
        #              f"\n")
        ###########################################################################
# we combine the dictionaries into data frames
eval_metrics_train = pd.DataFrame.from_dict(train_eval)
eval_metrics_test =  pd.DataFrame.from_dict(test_eval)
eval_metrics_test_FS52mix = pd.DataFrame.from_dict(FS52mix_eval)
eval_metrics_test_FS72mix = pd.DataFrame.from_dict(FS72mix_eval)
eval_metrics_test_mix2FS5 = pd.DataFrame.from_dict(mix2FS5_eval)
eval_metrics_test_mix2FS7 = pd.DataFrame.from_dict(mix2FS7_eval)
# and label the metrics according to the model
eval_metrics_train.columns = ['R2_mix_train', 'r_mix_train', 'MAE_mix_train', 'RMSE_mix_train']
eval_metrics_test.columns = ['R2_mix_test', 'r_mix_test', 'MAE_mix_test', 'RMSE_mix_test']
eval_metrics_test_FS52mix.columns = ['R2_FS52mix', 'r_FS52mix', 'MAE_FS52mix', 'RMSE_FS52mix']
eval_metrics_test_FS72mix.columns = ['R2_FS72mix', 'r_FS72mix', 'MAE_FS72mix', 'RMSE_FS72mix']
eval_metrics_test_mix2FS5.columns = ['R2_mix2FS5', 'r_mix2FS5', 'MAE_mix2FS5', 'RMSE_mix2FS5']
eval_metrics_test_mix2FS7.columns = ['R2_mix2FS7', 'r_mix2FS7', 'MAE_mix2FS7', 'RMSE_mix2FS7']
#
print("Mean and Standard Deviation for training and test data prediction evaluation metrics.")
singleframe = pd.concat([eval_metrics_train,eval_metrics_test, eval_metrics_test_FS52mix, eval_metrics_test_FS72mix, eval_metrics_test_mix2FS5, eval_metrics_test_mix2FS7],axis = 1)
sumstats = {"Mean": singleframe.mean(), "SD": singleframe.std()}
sumstats = pd.DataFrame(sumstats)
print(sumstats)
# save the performance frame
sumstats.to_csv(savepath+'shuffled_training_and_testing_sumstats.csv', sep=',')
# done
print("Elapsed time:")
end = timer()
print(timedelta(seconds=end-start))
print("########################")
print("Interpretation time!")
