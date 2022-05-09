FROM python:3.9-slim-bullseye

# installs buildtools and cmake, upgrades pip, and cleans up
RUN apt update -y &&\ 
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends \
        build-essential \
        cmake && \
    apt autoremove -y && \
    rm -rf /usr/local/src/* && \
    pip install --upgrade pip

# create non-root user environment to install our packages to
# this is also the user under which the ENTRYPOINT statement 
# will be executed
RUN useradd --create-home containeruser
USER containeruser
# add python user packages packages to PATH
ENV PATH="${PATH}:/home/containeruser/.local/bin"

# copy python code to image
WORKDIR /home/containeruser/
COPY app.py ./

# install python packages
RUN pip install --user --no-cache-dir flask lightgbm google-cloud-logging google-cloud-storage

# set up entry point for container
ENTRYPOINT ["python", "app.py"]
