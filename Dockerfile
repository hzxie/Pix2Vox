FROM manifoldai/orbyter-dl-dev

# check our python environment
RUN python3 --version
RUN pip3 --version

RUN pip install --upgrade pip

# set the working directory for containers
WORKDIR  /usr/src/2D-views-to-3D-objects

# install packages from file
RUN apt-get install libglib2.0-0
#RUN xargs -a packages.txt sudo apt-get install

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY models/ models/
RUN ls -la models/*

COPY core/ core/
RUN ls -la core/*

COPY utils/ utils/
RUN ls -ls utils/*
