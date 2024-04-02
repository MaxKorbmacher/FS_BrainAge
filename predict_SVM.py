# predict from trained models
# Max Korbmacher, 02.04.2024
print('Here, we predict age in the test data sets, based on SVM models trained on cortical volume, thickness and surface are extracted from FS5 and FS7.')
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import pandas as pd
import pickle
#Loading the saved model
with open('/Users/max/Documents/Projects/FS_brainage/FS5/results/SVM_model.pkl', 'rb') as pickle_file:
    FS5_model = pickle.load(pickle_file)
with open('/Users/max/Documents/Projects/FS_brainage/FS7/results/SVM_model.pkl', 'rb') as pickle_file:
    FS7_model = pickle.load(pickle_file)


# New data to predict
FS5 = pd.read_csv('/Users/max/Documents/Projects/FS_brainage/FS5/test.csv')
FS7 = pd.read_csv('/Users/max/Documents/Projects/FS_brainage/FS7/test.csv')

# define the columns which should be excluded for the prediction
pred_cols = FS5.columns[~FS5.columns.isin(['eid', 'Age', 'Sex', 'Scanner'])]

# apply the models to the data
FS5_FS5 = FS5_model.predict(FS5[pred_cols])
FS5_FS7 = FS5_model.predict(FS7[pred_cols])
FS7_FS7 = FS7_model.predict(FS7[pred_cols])
FS7_FS5 = FS7_model.predict(FS5[pred_cols])
Age1 = FS5['Age']
Age2 = FS7['Age']
eid1 = FS5['eid']
eid2 = FS7['eid']

# make a single df of all of it
df1=pd.DataFrame({'eid':eid1,'Age':Age1, 'FS5_FS5':FS5_FS5, 'FS5_FS7':FS5_FS7})
df2=pd.DataFrame({'eid':eid2,'Age':Age2, 'FS7_FS5':FS7_FS5,'FS7_FS7':FS7_FS7})

# now, save the df
df1.to_csv('/Users/max/Documents/Projects/FS_brainage/FS5predictions_SVM.csv', index=False)  
df2.to_csv('/Users/max/Documents/Projects/FS_brainage/FS7predictions_SVM.csv', index=False)  
print("All done.")