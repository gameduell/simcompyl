FROM continuumio/miniconda3

ADD envs/prod.yaml .
RUN conda update -n base -c defaults conda && conda env create -f prod.yaml

ENV PATH /opt/conda/envs/sim-prod/bin:$PATH
ENV CONDA_DEFAULT_ENV sim-prod

ADD setup.py README.rst requirements.txt /modules/simcompyl/
ADD simcompyl /modules/simcompyl/simcompyl

RUN cd /modules/simcompyl \
    && python3 setup.py develop --no-deps --optimize 2

RUN groupadd -r py && useradd --no-log-init -r -m -g py py
USER py:py

