#Importing Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

#Loading Model
model = keras.models.load_model("C:/Users/Godwin/Downloads/xception_v1_03_0.868.h5")

#loading img
path = "C:/Users/Godwin/Documents/Workflow/MLZoomcamp/clothing-dataset-small/train/"
name = "pants/1d407629-87e5-4702-b9fc-e3b19de93570.jpg"
img_path = path + name

img = load_img(img_path, target_size= (299,299,3))
img = np.array(img)
x = np.array([img])
x = preprocess_input(x)

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
#making predictions
pred = model.predict(x)
out = dict(zip(classes, pred[0]))

#Converint to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite = converter.convert()
with open('lite_model.tflite', 'wb') as f:
    f.write(tflite)


def prediction(path, img):

    interpreter = tflite.Interpreter(model_path = path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    out = dict(zip(classes, preds[0]))
    return out

def preprocessing(path):
    with Image.open(path) as img:
        img = img.resize((299,299), Image.NEAREST)

    img = np.array(img, dtype = 'float32')
    img = np.array([img])
    img/= 127.5
    img -= 1
    return img

image = preprocessing(img_path)
pred = prediction("C:/Users/Godwin/Documents/Workflow/MLZoomcamp/ML-ZoomCamp/Week9/lite_model.tflite" , image)