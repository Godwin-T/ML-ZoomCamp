from PIL import Image
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy as np

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

interpreter = tflite.Interpreter(model_path = "lite_model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

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

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambdahandler(event, context):
    url = event['url']
    result = predict(url)
    return result