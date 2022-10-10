from flask import Flask, request, jsonify
import waitress
import sklearn
import pickle

file = 'model1.bin'
def read(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)

    return file
dv = read('dv.bin')
model = read('model1.bin')

app = Flask(__name__)
@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()

    df = dv.transform([customer])
    prediction = model.predict_proba(df)[0,1]
    churn_decision = prediction >= 0.5

    result = {'Prediction': float(prediction), 'Churn Decision': bool(churn_decision)}
    return jsonify(result)

if __name__ == '__main__':
    waitress.serve(app, port = 5000)