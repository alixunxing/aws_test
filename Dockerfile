FROM python:3.9

RUN apt-get update
RUN apt-get -y install python3-dev
# RUN apt-get -y install libleptonica-dev tesseract-ocr-dev libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn

RUN pip install --upgrade pip setuptools

RUN mkdir /app
COPY ./data /app
COPY ./mnist_cnn.py /app
COPY ./requirements.txt /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT python mnist_cnn.py
