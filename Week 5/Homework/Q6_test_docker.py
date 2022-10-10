import requests

url = 'http://127.0.0.1:5000/predict'
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
output = requests.post(url, json=client).json()
print(output)
