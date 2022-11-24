FROM python:3.9-slim-buster as serve

USER root
RUN mkdir -p /source
RUN useradd -ms /bin/bash -g root dra-user
RUN chown dra-user /source

COPY . /source


RUN apt-get update -y && \
  apt-get install -y build-essential && \
  apt-get install -y curl

WORKDIR /source

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
  && python get-pip.py && rm get-pip.py

RUN make install

USER dra-user

FROM serve as test

USER root
RUN make install-all

USER dra-user
ENTRYPOINT ["make", "test"]
