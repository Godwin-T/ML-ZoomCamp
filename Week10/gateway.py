import warnings
warnings.filterwarnings('ignore')

import os
import grpc
import tensorflow as tf
from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask, jsonify
from flask import request

from proto import np_to_protobuf


host = os.getenv('TF-SERVING', 'localhost:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
preprocessor =  create_preprocessor('xception', target_size= (299,299))

def process_request(data):

    #url  = 'http://bit.ly/mlbookcamp-pants'
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(data))
    return pb_request

classes = ['dress',
            'hat',
            'longsleeve',
            'outwear',
            'pants',
            'shirt',
            'shoes',
            'shorts',
            'skirt',
            't-shirt']

def process_response(request):
    
    pred = request.outputs['dense_7'].float_val
    out = dict(zip(classes, pred))
    return out
def prediction(url):
    data = preprocessor.from_url(url)
    request = process_request(data)
    pb_response = stub.Predict(request, timeout= 20.0)
    pred = process_response(pb_response)
    return pred

app = Flask('gateway')
@app.route("/predict", methods = ['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    pred = prediction(url)
    result = jsonify(pred)
    return result


if __name__ == '__main__':
    #url = 'http://bit.ly/mlbookcamp-pants'
    #pred = prediction(url)
    #print(pred)
    app.run(debug=True, host = '0.0.0.0', port = 9696)
    #serve(app, port=5000)