
#Importing Libraries
print("Importing Libraries")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import pickle
import seaborn as sns
from IPython.display import display
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_curve, auc, roc_auc_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

#Parameters
out = 'Churn.bin'

print("Data Preparation")
#Data Preparation
data = pd.read_csv("C:/Users/Godwin/Documents/Workflow/ML Zoomcamp/Classification/Customer-churn/Telco-Customer-Churn.csv")
data.columns = data.columns.str.replace(' ', '_').str.lower()

categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()

for col in categorical_col:
    data[col] = data[col].str.replace(' ', '_').str.lower()



data['totalcharges'] = pd.to_numeric(data['totalcharges'], errors= 'coerce')
data['totalcharges'].fillna(data['totalcharges'].mean(), inplace = True)
data['churn'] = (data.churn=='yes').astype(int)
categorical_col = data.dtypes[data.dtypes == 'object'].index.tolist()
numerical_col = ['tenure', 'totalcharges', 'monthlycharges']

categorical_col.remove('customerid')

#Metric Funtion
def metric(actual, predicted, t):

    accuracy = (predicted == actual).mean()
    actual_positive = (actual == 1)
    actual_negative = (actual == 0)

    predicted_positive = (predicted >= t)
    predicted_negative = (predicted < t)



    tp = (actual_positive & predicted_positive).sum()
    tn = (actual_negative & predicted_negative).sum()
    fp = (actual_negative & predicted_positive).sum()
    fn = (actual_positive & predicted_negative).sum()

    tpr = tp/ (tp + fn)
    fpr = fp/ (fp + tn)

    precision = tp/(tp + fp)
    recall = tp/(tp +fn)
    f1_score = 2 * ((precision * recall)/ (precision + recall))

    return tn, fp, fn, tp, precision, recall, tpr, fpr, f1_score#, accuracy

#Vectorizer
dv = DictVectorizer(sparse = False)

#Define train function
def train(data, y, c):
    dv.fit(data[categorical_col + numerical_col].to_dict(orient = 'records'))
    X_train = dv.transform(data[categorical_col + numerical_col].to_dict(orient = 'records'))

    model = LogisticRegression(C = c, max_iter = 1000)
    model.fit(X_train, y)
    return dv, model

#Define predict function
def predict(data, dv, model):
    X_test = dv.transform(data[categorical_col + numerical_col].to_dict(orient = 'records'))
    prediction = model.predict_proba(X_test)[:,1]
    return prediction

#Spliting Data
print("Spliting the data")
train_data,test_data = train_test_split(data, test_size = 0.2, random_state = 1)

y_train = train_data.pop('churn')
y_test = test_data.pop('churn')

print('Training the model')
#Training Model
dv, model = train(train_data, y_train, c = 1)
prediction = predict(test_data, dv, model)

tn, fp, fn, tp, precision, recall, tpr, fpr, f1_score = metric(y_test, prediction, 0.5)
cm = np.array([[tn, fp], [fn, tp]])
print(f' The auc score of the model prediction is {roc_auc_score(y_test,prediction)}')

#Saving model
with open(out, 'wb') as f:
    pickle.dump((dv,model), f)

print(f'The model has been save to {out}')