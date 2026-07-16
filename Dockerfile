FROM python:3.12-slim-bookworm
#@sha256:28cf028e5a544e92dbe11450debd93dd5eb70eaf3179a9e878cfaee426556b3b

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
    libhdf5-serial-dev \
    libffi-dev \
    netcdf-bin \
    libnetcdf-dev

RUN python3 -m pip install --upgrade setuptools pip wheel

# Check python & pip
RUN python --version
RUN which python
RUN pip --version
RUN which pip

COPY . /pyfv3

# Install pyFV3 and the full dependencies
RUN cd /pyfv3 && pip install -e .[dev]

RUN pip install \
    matplotlib \
    cython \
    cartopy \
    ipyparallel \
    jupyter \
    jupyterlab \
    shapely \
    jupyterlab_code_formatter \
    mpi4py \
    pytest \
    pytest-subtests \
    pytest-regressions \
    pytest-profiling \
    pytest-cov

# # set up for fv3viz
RUN cd / && \
    git clone https://github.com/oelbert/fv3viz

RUN python -m ensurepip --upgrade && \
    python -m pip install \
    /fv3viz

RUN python -m pip install pybind11==2.13.6

ENV PYTHONPATH=/fv3viz:/pace/external/gt4py/src

ENV CFLAGS="-I/usr/include -DACCEPT_USE_OF_DEPRECATED_PROJ_API_H=1"

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
