import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://4aioe5pxga.execute-api.us-east-1.amazonaws.com/Test/predict'
data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)