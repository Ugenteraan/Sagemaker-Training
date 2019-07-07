#sagemaker-horus-category-serve
FROM ubuntu:16.04
MAINTAINER ugenteraan@ombre.app

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get upgrade -y
RUN apt-get install wget -y
RUN apt-get install python3.5 python3.5-dev -y
RUN apt-get update --fix-missing
RUN apt-get install python3-pip libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6 libgl1-mesa-glx -y
RUN pip3 install pip --upgrade
RUN pip3 install tensorflow==1.9.0 boto3
RUN pip3 install opencv-python requests h5py
COPY cred.py /home
COPY model.py /home
COPY train /home
RUN chmod +x /home/train
RUN mkdir -p /opt/ml/data/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
#ENV PATH="/opt/program:${PATH}"

WORKDIR /home

ENTRYPOINT ["/usr/bin/python3", "/home/train"]

EXPOSE 8080 