import requests

url = "http://127.0.0.1:9696/predict"
data = {"customerid": "7892-pookp",
        "gender": "female",
        "seniorcitizen": 0,
        "partner": "yes",
        "dependents": "no",
        "tenure": 28,
        "phoneservice": "yes",
        "multiplelines": "yes",
        "internetservice": "fiber_optic",
        "onlinesecurity": "no",
        "onlinebackup": "no",
        "deviceprotection": "yes",
        "techsupport": "yes",
        "streamingtv": "yes",
        "streamingmovies": "yes",
        "contract": "month-to-month",
        "paperlessbilling": "yes",
        "paymentmethod": "electronic_check",
        "monthlycharges": 104.8,
        "totalcharges": 3046.05}
response = requests.post(url, json = data).json()
print(response)

if response['Churn'] == True:
    print(f"The customer with the an ID of {data['customerid']} probability of churning is {response['Prediction']}")