#Loading Model
import sklearn
import pickle
from flask import Flask, jsonify
from flask import request
from waitress import serve

out = 'Churn.bin'
with open(out, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('Churn')
@app.route("/predict", methods = ['POST'])
def predict():

    customer = request.get_json()
    df = dv.transform([customer])
    prediction = model.predict_proba(df)[0,1]
    churn = prediction >= 0.5

    results = {'Prediction':float(prediction), 'Churn':bool(churn)}

    return jsonify(results)

@app.route("/")
def index():
    return "Homepage of GeeksForGeeks"


if __name__ == '__main__':
    serve(app, port=5080)