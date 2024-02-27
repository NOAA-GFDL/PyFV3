from ndsl.constants import CONST_VERSION, ConstantVersions


# By selecting the GEOS constant using (PACE_CONSTANT env var)
# we select the GEOS flavor of numerics
IS_GEOS = ConstantVersions.GEOS == CONST_VERSION
