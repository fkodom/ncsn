FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

WORKDIR /usr/src/app

COPY .git/ .git/
COPY ncsn/ ncsn/
COPY *.py ./
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN mkdir data models mlruns media

EXPOSE 5000
