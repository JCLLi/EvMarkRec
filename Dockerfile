FROM tensorflow/tensorflow:latest-gpu

WORKDIR /EvMarkRec

RUN pip install opencv-python

RUN apt-get update && apt-get install -y libgl1-mesa-glx

