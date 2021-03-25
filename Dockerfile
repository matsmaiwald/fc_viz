# syntax = docker/dockerfile:experimental
FROM python:3.8-buster
# install build utilities
RUN apt-get update && \
    apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/fc_viz

# Installing python dependencies
COPY requirements_general.txt .
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --no-cache-dir -r requirements_general.txt

COPY requirements.txt .
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

# Running Python Application
CMD ["python3", "/src/fc_poc.py"]