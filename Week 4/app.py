import sklearn
import pickle
from flask import Flask, request,render_template


out = 'Class/Churn.bin'
with open(out, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask(__name__)
@app.route('/')

def running():
    return render_template("index.html")


# def submit():
#     if request.method == 'POST':
#         name = request.form['username']

#     return render_template('sub.html', n = name)
@app.route('/sub', methods = ['POST'])
def run():
    if request.method == 'POST':
        data = {}
        data["customerid"] = request.form["customerid"]
        data["gender"] = request.form["gender"]
        data["seniorcitizen"] = request.form["seniorcitizen"]
        data["partner"] = request.form["partner"]
        data["dependents"] = request.form["dependent"]
        data["tenure"] = int(request.form["tenure"])
        data["phoneservice"] = request.form["phoneservice"]
        data["multiplelines"] = request.form["multiplelines"]
        data["internetservice"] = request.form["internetservice"]
        data["onlinesecurity"] = request.form["onlinesecurity"]
        data["onlinebackup"] = request.form["onlinebackup"]
        data["deviceprotection"] = request.form["deviceprotection"]
        data["techsupport"] = request.form["techsupport"]
        data["streamingtv"] = request.form["streamingtv"]
        data["streamingmovies"] = request.form["streamingmovies"]
        data["contract"] = request.form["contract"]
        data["paperlessbilling"] = request.form["paperlessbilling"]
        data["paymentmethod"] = request.form["paymentmethod"]
        data["monthlycharges"] = int(request.form["monthlycharges"])
        data["totalcharges"] = int(request.form["totalcharges"])


    df = dv.transform(data)
    prediction = model.predict_proba(df)[0,1]
    churn = prediction >= 0.5

    results = {'Prediction':float(prediction), 'Churn':bool(churn)}

    return render_template('sub.html', n = results)

if __name__ == "__main__":
    app.run(debug=True)
#     df = dv.transform(data)
#     prediction = model.predict_proba(df)[0,1]
#     churn = prediction >= 0.5

#     results = {'Prediction':float(prediction), 'Churn':bool(churn)}

