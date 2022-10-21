import wget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('transformed_data.csv')

cat_col = ['home','marital','records', 'job']
num_col = ['seniority', 'time', 'age', 'expenses', 
            'income', 'assets', 'debt', 'amount', 'price']

    
full_train_df, full_test_df = train_test_split(df, test_size =0.2, random_state=11)


y_train = (full_train_df['status'] == 'default').astype('int')
y_test = (full_test_df['status'] == 'default').astype('int')


dv = DictVectorizer(sparse = False)
dv.fit(full_train_df[cat_col + num_col].to_dict(orient = 'records'))
feature_names = dv.get_feature_names()


X_train = dv.transform(full_train_df[cat_col + num_col].to_dict(orient = 'records'))
X_test = dv.transform(full_test_df[cat_col + num_col].to_dict(orient = 'records'))


del full_train_df['status']
del full_test_df['status']


dtrain = xgb.DMatrix(X_train, label = y_train, feature_names = feature_names)
dtest = xgb.DMatrix(X_test, feature_names = feature_names)


x_params = {
    'eta': 0.1,
    'max_depth':10,
    'min_child_weight':30,

    'objective':'binary:logistic',
    'n_threads':8,

    'seed':1,
    'verbosity':0
}

model = xgb.train(x_params, dtrain = dtrain,num_boost_round = 125)
prediction = model.predict(dtest)
auc = roc_auc_score(y_test, prediction)
print(auc)

def save_file(filename, file):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)
    print('File saved')

encoder = 'dv.bin'
algorithm = 'model.bin'
col_names = 'features.bin'
encoder = save_file(encoder, dv)
model = save_file(algorithm, model)
features = save_file(col_names, feature_names)