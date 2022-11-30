#Importing Libraries
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite


#loading the model    
interpreter = tflite.Interpreter(model_path = 'dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

#checking indexes
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

#Download image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

#preprocess image
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prediction(url):

    img = download_image(url)
    img = prepare_image(img, (150,150))
    img = np.array(img, dtype = 'float32')
    img = np.array([img])
    img = img/255.

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    out = {'value': float(preds[0, 0])}
    return out

def lambdahandler(event, context):
    path = event['url']
    result = prediction(path)
    return result
