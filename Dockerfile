FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

MAINTAINER lachiew <whitehead@wehi.edu.au>
WORKDIR /denoising/
COPY requirements.txt requirements.txt

RUN apt update \
	&& apt install \
		fontconfig \
		python3-pyqt5 -y

RUN pip install --upgrade pip \ 
	&& pip install -r requirements.txt
