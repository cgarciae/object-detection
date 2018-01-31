FROM tensorflow/tensorflow:1.5.0-gpu-py3

RUN apt-get update
RUN apt-get install -y git curl

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash \
    && apt-get install -y nodejs

RUN pip install -U git+https://github.com/jupyterlab/jupyterlab@v0.31.2


RUN pip install lxml
RUN pip install xmltodict
RUN pip install python_path
RUN pip install plotly
RUN jupyter labextension install @jupyterlab/plotly-extension
RUN pip install dicto
RUN pip install imgaug
RUN pip install opencv-python

RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev

RUN pip install cytoolz

RUN pip install -U dicto
RUN pip install click

RUN apt-get install -y wget

RUN apt-get install -y python3-tk

ENV PYTHONPATH $PYTHONPATH:/code:/code/slim