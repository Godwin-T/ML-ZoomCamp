from flask import Flask, jsonify, request
import sklearn
import xgboost as xgb
import pickle

def load_file(filename):
    with open(filename, 'rb') as f:
        out = pickle.load(f)
    return out

dv = load_file('dv.bin')
model = load_file('model.bin')
feature = load_file('features.bin')

app = Flask(__name__)
@app.route('/predict', methods = ['POST'])

def prediction():

    customer = request.get_json()

    data = dv.transform([customer])
    data = xgb.DMatrix(data, feature_names = feature)
    prediction = model.predict(data)
    out = (float(str(prediction).strip('[]')))
    return jsonify(out)

@app.route("/")
def index():
    return "Homepage of GeeksForGeeks"

if __name__ == "__main__":
    app.run(debug = True)
