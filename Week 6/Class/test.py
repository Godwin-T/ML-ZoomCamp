import sklearn
import xgboost as xgb
import pickle

def load_file(filename):
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return(out)

dv = load_file('dv.bin')
model = load_file('model.bin')
features = load_file('features.bin')

data = {'Unnamed: 0': 3242,
        'seniority': 2,
        'home': 'owner',
        'time': 36,
        'age': 37,
        'marital': 'married',
        'records': 'no_rec',
        'job': 1,
        'expenses': 60,
        'income': 125.0,
        'assets': 2000.0,
        'debt': 0.0,
        'amount': 450,
        'price': 1490}

data = dv.transform(data)
data = xgb.DMatrix(data,feature_names = features)
pred = model.predict(data)
print(float(str(pred).strip('[]')))