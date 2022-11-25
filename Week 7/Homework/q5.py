import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

    
model = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
#dv = dv = model.custom_objects['dictvectorizer']
runner = model.to_runner()

svc = bentoml.Service('credict_risk_classifier_rf', runners=[runner])

@svc.api(input=NumpyNdarray(), output=JSON())
def classify(data):

    #data = credict_application_data.dict()
    #n_data = dv.transform(data)
    prediction = runner.predict.run(data)
    print(prediction)

    result = prediction[0]
    if result >=0.5:
        return {'Status': "Declined"}

    elif result >=0.25:
        return {'Status': "Under-Consideration"}
    else:
        return{'status': 'Approved'}
