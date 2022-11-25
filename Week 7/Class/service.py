import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class CreditApplication(BaseModel):
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


model = bentoml.xgboost.get("credict_risk:meupd6spxggv4aav")
dv = model.custom_objects['dictvectorizer']
runner = model.to_runner()
svc = bentoml.Service('credict_risk_classifier', runners=[runner])

@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
async def classify(credict_application):

    data = credict_application.dict()
    n_data = dv.transform(data)
    prediction = await runner.predict.async_run(n_data)
    print(prediction)

    result = prediction[0]
    if result >=0.5:
        return {'Status': "Declined"}

    elif result >=0.25:
        return {'Status': "Under-Consideration"}
    else:
        return{'status': 'Approved'}
