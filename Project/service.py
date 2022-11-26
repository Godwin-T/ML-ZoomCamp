import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class RainPrediction(BaseModel):
    date: str
    location: str
    rainfall: float
    evaporation: float
    sunshine: float
    windgustdir: str
    windgustspeed: float
    winddir9am: str
    winddir3pm: str
    windspeed9am: float
    windspeed3pm: float
    humidity9am: float
    humidity3pm: float
    pressure9am: float
    pressure3pm: float
    cloud9am: float
    cloud3pm: float
    temp9am: float
    raintoday: str


model = bentoml.xgboost.get("rainfall_prediction:gultljthiwvzsaav")
dv = model.custom_objects['dictvectorizer']
runner = model.to_runner()
svc = bentoml.Service('Rainfall_Prediction', runners=[runner])

@svc.api(input=JSON(pydantic_model=RainPrediction), output=JSON())
async def classify(data):

    data = data.dict()
    n_data = dv.transform(data)
    prediction = await runner.predict.async_run(n_data)
    print(prediction)

    result = prediction[0]
    if result < 0.5:
        output = "There won't be rain tomorrow"
        return{'status': output}
    else:
        output = "It will rain tomorrow"
        return{'status': output}