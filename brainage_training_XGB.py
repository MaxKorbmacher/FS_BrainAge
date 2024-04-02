#Author: Ann-Marie de Lange (a.m.g.d.lange@psykologi.uio.no)
# Adapted by Max Korbmacher (max.korbmacher@gmail.com)
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.lines as mlines
from scipy import stats
#from imblearn.under_sampling import RandomUnderSampler
#from sklearn.inspection import plot_partial_dependence
#from pdpbox import pdp, get_dataset, info_plots
import sys, os
import statsmodels.api as sm
import json

def pearsonr_ci(x,y,alpha=0.05):

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

import seaborn as sns
sns.set(color_codes=True)
sns.set(font_scale=2)
sns.set_style("white")

#Using this to get LaTeX font for plots (LaTeX code rules must be followed for e.g. axis titles (no underscores etc without \))
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)


####################################

###### ADD PATHS TO DATA AND FOLDERS WHERE FILES WILL BE SAVED
datapath="/Users/max/Documents/Projects/FS_brainage/FS5/"
savepath="/Users/max/Documents/Projects/FS_brainage/FS5/results/"


###### USE DICTIONARIES TO LOOP OVER ALL DIFFUSION MODELS AND RUN BRAIN AGE PREDICTION
dict = {}
corr = {}
pred = {}


#Output dictionary to store r and CI in json (to use for LaTeX table in step 8)
out_dict = {}

# DEFINE MODELS
model = ["train"]

for m in model:
    #saves the results to text files
    file=savepath+'Output_XGB_%s.txt'%m
    with open(file, 'w') as text_file:

        text_file.write("===============\n")
        text_file.write("MODEL = %s\n" %m)
        text_file.write("===============\n")

        dict['%s' % m] = pd.read_csv(datapath+'%s.csv' %m)

        #To test run with smaller N:
        #dict['%s' % m] = dict['%s' % m].sample(frac=0.01)


        # CHECK FILE CONTENT AND LENGTH
        print ('printing head of for %s' %m)
        print (dict['%s' % m].head(5))
        print ('printing number of columns for %s' %m)
        print (len(dict['%s' % m].columns))
        print ('printing length of datafile for %s' %m)
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
        print ('printing final x for %s' % m)
        print (x.head(5))
        print ('printing final y for %s' % m)
        print (y.head(5))


        # SPECIFY MODEL

        # configure cross-validation procedure
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

        # define the model
        # the presented hyperparameters were defined based on previous multimodal brain age predictions.
        #search = xgb.XGBRegressor(seed=42)
        search = xgb.XGBRegressor(objective= 'reg:squarederror',nthread=4,seed=42)

            
        text_file.write ('validating %s model\n' % m)

        text_file.write ('------------------------------\n')
        text_file.write ('RMSE values:\n')
        RMSE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_root_mean_squared_error',n_jobs = 4)
        text_file.write('Mean and STD for RMSE: %.3f (%.3f)\n' % (mean(RMSE), std(RMSE)))

        text_file.write ('------------------------------\n')
        text_file.write ('MAE values:\n')
        MAE = cross_val_score(search, x, y, cv=cv_outer,scoring='neg_mean_absolute_error',n_jobs = 4)
        text_file.write('Mean and STD for MAE: %.3f (%.3f)\n' % (mean(MAE), std(MAE)))

        text_file.write ('------------------------------\n')
        text_file.write ('R2 values:\n')
        R2 = cross_val_score(search, x, y, cv=cv_outer,scoring='r2',n_jobs = 4)
        text_file.write('Mean and STD for R2: %.3f (%.3f)\n' % (mean(R2), std(R2)))
            


        # RUN CROSS_VAL_PREDICT
        print ('Running cross_val_predict')

        # DEFINE THE VARIABLES PRED (PREDICTED AGE) AND BAG (BRAIN AGE GAP)
        pred[m] = cross_val_predict(search, x, y, cv=cv_outer, n_jobs=2)
        BAG = pred[m] - y


        # ADD PREDICTED BRAIN AGE AND BRAIN AGE GAP TO X_COPY TO GET A FULL DATAFRAME WITH ALL VARIABLES
        x_copy['pred_age_%s' % m] = pred[m]
        x_copy['brainage_gap_%s' % m] = x_copy['pred_age_%s' % m] - y


        # RUN CORRELATION FOR PREDICTED VERSUS TRUE AGE
        text_file.write ('running pearsons corr for predicted versus true age for %s\n' % m)

        corr[m] = pearsonr_ci(x_copy['Age'],x_copy['pred_age_%s' %m])
        text_file.write ("r = %s\n" % corr[m][0])
        text_file.write ("r p-value = %s\n" % corr[m][1])
        text_file.write ("r CI = [%s,%s]\n" % (corr[m][2],corr[m][3]))

        text_file.write ('------------------------------\n')


        # CALCULATE BAG RESIDUALS (THE BAG VALUES RESIDUALISED FOR CHRONOLOGICAL AGE)
        print ('fitting data for age correction for %s' % m)
        z = np.polyfit(y, BAG, 1)
        resid = BAG - (z[1] + z[0]*y)
        x_copy["BAG_residual_%s" % m] = resid
        print (x_copy.head(5))
        
        # CALCULATE THE CORRECTED BAG
        correct = np.polyfit(y,pred[m], 1)
        BAG_COR = pred[m] + (y - (correct[0]*y + correct[1]))
        x_copy["BAG_corrected_%s" % m] = BAG_COR


        # SAVE FILE
        # First create a dataframe including only relevant variables
        x_copy_save = x_copy[['eid','Age','pred_age_%s' % m,'brainage_gap_%s' % m,'BAG_residual_%s' % m, 'BAG_corrected_%s' % m]]
        #print (x_copy_save.head(5))	#uncomment to check file content
        print ('saving file with brain age estimates')
        x_copy_save.to_csv(savepath+'Brainage_XGB_%s.csv' %m, sep=',',index=None)

        #Store r and CI in dict for writing to json
        out_dict[m] = corr[m]


        #Write the JSON with r and CI values to json
        #with open(savepath+'r_vals_%s.json' %m,'w') as f:
        #    json.dump(out_dict, f, sort_keys=True, indent=4)
        # fit the model
        result = search.fit(x,y)
        # Save the model
        result.save_model(savepath+'XGB_model.txt')
        # estimate feature gain (one metric for feature importance)
        feature_importances = result.get_booster().get_score(importance_type='gain')

        keys = list(feature_importances.keys())
        values = list(feature_importances.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        data.to_csv(savepath+'FIXED_Feature_Importances_%s.csv' %m, sep=',')
        print('Estimate permutation feature importance.')
        r = permutation_importance(result, x, y,
                                   n_repeats=100,
                                   random_state=0,
                                   n_jobs = 4)
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                text_file.write(f"{x.columns[i]:<8} "
                      f"{r.importances_mean[i]:.6f}"
                      f" +/- {r.importances_std[i]:.6f}"
                      f"\n")

