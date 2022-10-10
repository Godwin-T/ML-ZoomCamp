import sklearn
import pickle

def load(filename):
    with open(filename, 'rb') as f:
        output = pickle.load(f)

    return output

dv = load('dv.bin')
model = load('model1.bin')

data = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
data_transf = dv.transform([data])

prediction = model.predict_proba(data_transf)[0,1]
print(prediction)


    
