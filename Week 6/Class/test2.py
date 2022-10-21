import requests

url = 'http://127.0.0.1:5000/predict'
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

response = requests.post(url, json = data).json()
print(response)
