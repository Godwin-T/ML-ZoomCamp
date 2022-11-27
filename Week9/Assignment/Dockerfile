FROM public.ecr.aws/lambda/python:3.9

COPY lite_dino-dragon_model.tflite .

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl