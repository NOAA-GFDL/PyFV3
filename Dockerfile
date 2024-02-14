FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt install -y --no-install-recommends \
    software-properties-common

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends\
    g++ \
    gcc \
    gfortran \
    git \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libopenmpi3 \
    libopenmpi-dev \
    libboost-all-dev \
    libhdf5-serial-dev \
    netcdf-bin \
    libnetcdf-dev \
    curl \
    patch \
    libreadline-dev \
    build-essential

# RUN apt-get update -y && \
#     apt install -y --no-install-recommends \
#     python3.8 \
#     python3.8-dev \
#     python3-pip

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 60

# Install py from source via pyenv manager
RUN git clone --recursive https://github.com/pyenv/pyenv.git .pyenv
ENV PATH=".pyenv/shims:.pyenv/bin:${PATH}"
ENV PYENV_ROOT=".pyenv"
ENV PYTHON_VERSION=3.8.12
RUN pyenv install ${PYTHON_VERSION}
RUN pyenv global ${PYTHON_VERSION}

RUN python3 -m pip install --upgrade setuptools pip wheel

# Check python & pip
RUN python --version
RUN which python
RUN pip --version
RUN which pip

COPY . /pyFV3

RUN cd /pyFV3 && \
    pip install -r /pyFV3/requirements.txt

RUN pip install \
    matplotlib \
    cython \
    cartopy \
    ipyparallel \
    jupyter \
    jupyterlab \
    shapely \
    cartopy \
    jupyterlab_code_formatter

# # set up for fv3viz
# RUN cd /
# RUN git clone https://github.com/ai2cm/fv3net.git
# RUN cd fv3net && git checkout 1d168ef
# RUN pip install fv3net/external/vcm
# ENV PYTHONPATH=/fv3net/external/fv3viz

RUN git clone --recursive https://github.com/NOAA-GFDL/NDSL.git
RUN cd /NDSL && git checkout 75181f4
RUN cd /NDSL && pip install -e .

ENV CFLAGS="-I/usr/include -DACCEPT_USE_OF_DEPRECATED_PROJ_API_H=1"

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
