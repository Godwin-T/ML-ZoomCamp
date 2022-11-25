import numpy as np
import tensorflow.lite as tflite
from PIL import Image

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

def preprocessing(path):
    with Image.open(path) as img:
        img = img.resize((299,299), Image.NEAREST)

    img = np.array(img, dtype = 'float32')
    img = np.array([img])
    img/= 127.5
    img -= 1
    return img

def prediction(img_path):

    img = preprocessing(img_path)
    interpreter = tflite.Interpreter(model_path = "C:/Users/Godwin/Documents/Workflow/MLZoomcamp/ML-ZoomCamp/Week9/lite_model.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    out = dict(zip(classes, preds[0]))
    return out

def lambdahandler(event, context):
    path = event['url']
    result = prediction(path)
    return result