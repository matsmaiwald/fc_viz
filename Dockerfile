# syntax = docker/dockerfile:experimental
FROM python:3.8-buster
# install build utilities
RUN apt-get update && \
    apt-get install -y gcc g++ python-dev python3-dev make apt-transport-https ca-certificates build-essential zsh

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/fc_viz

# Installing python dependencies
COPY requirements_general.txt .
RUN pip install -r requirements_general.txt

COPY requirements_prophet.txt .
RUN pip install -r requirements_prophet.txt
# RUN pip install holidays tqdm
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all the files from the project’s root to the working directory
# COPY src/ /src/
# RUN ls -la /src/*

# Running Python Application
# CMD ["python3", "/src/fc_poc.py"]
CMD ["/bin/zsh"]