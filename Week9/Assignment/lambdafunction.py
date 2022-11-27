#Importing Libraries
import numpy as np
import wget
import keras
import tensorflow as tf
from io import BytesIO
from urllib import request
from PIL import Image

#Downloading model
link = 'https://github.com/SVizor42/ML_Zoomcamp/releases/download/dino-dragon-model/dino_dragon_10_0.899.h5'
def model_downloaer_and_converter(link):
    name = 'model.h5'
    wget.download(link, name)

    #Converting model to tflite
    model = keras.models.load_model('model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()

    with open('lite_dino-dragon_model.tflite', 'wb') as f:
        f.write(tflite)
    print("Model successfully converter")

#loading the model    
interpreter = tf.lite.Interpreter(model_path = 'lite_dino-dragon_model.tflite')
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
    return preds[0]

def lambdahandler(event, context):
    path = event['url']
    result = prediction(path)
    return result

#loading image
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'
