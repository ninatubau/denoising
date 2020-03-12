FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

MAINTAINER lachiew <whitehead@wehi.edu.au>
WORKDIR /denoising/
COPY requirements.txt requirements.txt

RUN apt update
RUN pip install --upgrade pip
RUN apt install fontconfig -y
RUN apt install python3-pyqt5 -y
RUN pip install -r requirements.txt
