#!/bin/bash

THIS_DIR=$PWD
TEST_DATA_PATH="../../../../test_data/geos/TEMP_XPPM_YPMM"
mkdir -p $TEST_DATA_PATH
cd $TEST_DATA_PATH

wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/YPPM-In.nc
wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/YPPM-Out.nc
wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/XPPM-In.nc
wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/XPPM-Out.nc
wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/input.nml
wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/translate/11.5.2/x86_GNU/Dycore/TBC_C24_L72_Debug/Grid-Info.nc


cd $THIS_DIR
rm -r ./.gt_cache_*

export PACE_FLOAT_PRECISION=32
export PACE_CONSTANTS=GEOS
export FV3_DACEMODE=Python

python -m pytest -v -s -x \
    --data_path=$TEST_DATA_PATH \
    --backend=numpy \
    --which_modules=XPPM,YPPM \
    --multimodal_metric \
    ../../../savepoint
