from PIL import Image
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy as np
import tensorflow.lite as tflite

#Classes
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
preprocessor  = create_preprocessor('xception', target_size=(299,299))

interpreter = tflite.Interpreter(model_path = "./lite_model.tflite")
interpreter.allocate_tensors()
def preprocessing(path):
    with Image.open(path) as img:
        img = img.resize((299,299), Image.NEAREST)
    img = np.array(img, dtype = 'float32')
    img = np.array([img])
    img/= 127.5
    img -= 1
    return img

def predict(url):
    X = preprocessor.from_url(url)

    img = preprocessor.from_url(url)
    #img = preprocessing(img_path)
    #interpreter = tflite.Interpreter(model_path = "C:/Users/Godwin/Documents/Workflow/MLZoomcamp/ML-ZoomCamp/Week9/lite_model.tflite")
    #interpreter.allocate_tensors()

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambdahandler(event, context):
    url = event['url']
    result = predict(url)
    return result