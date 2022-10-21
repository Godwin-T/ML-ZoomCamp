import bentoml
from bentoml.io import JSON

from pydantic import BaseModel

class UserProfile(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: int
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int
    
model = bentoml.sklearn.get("credict_risk_rf:arbrefcrfkdkuaav")
dv = dv = model.custom_objects['dictvectorizer']
rf_runner = model.to_runner()

svc = bentoml.Service('credict_risk_classifier_rf', runners=[rf_runner])

@svc.api(input=JSON(pydantic_model=UserProfile), output=JSON())
def classify(credict_application_data):

    data = credict_application_data.dict()
    n_data = dv.transform(data)
    prediction = rf_runner.predict.run(n_data)
    print(prediction)

    result = prediction[0]
    if result >=0.5:
        return {'Status': "Declined"}

    elif result >=0.25:
        return {'Status': "Under-Consideration"}
    else:
        return{'status': 'Approved'}
