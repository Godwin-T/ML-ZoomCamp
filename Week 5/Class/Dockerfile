FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "Churn.bin", "./"]

EXPOSE 9696

#ENTRYPOINT ['predict:app']