import bentoml
from bentoml.io import JSON

model = bentoml.xgboost.get("credict_risk:ekvvbesogogeacar")
dv = model.custom_objects['dictvectorizer']
runner = model.to_runner()
svc = bentoml.Service('credict_risk_classifier', runners=[runner])

@svc.api(input=JSON(), output=JSON())
def classify(data):
    n_data = dv.transform(data)
    prediction = runner.predict.run(n_data)
    print(prediction)

    result = prediction[0]
    if result >=0.5:
        return {'Status': "Declined"}

    elif result >=0.25:
        return {'Status': "Under-Consideration"}
    else:
        return{'status': 'Approved'}
