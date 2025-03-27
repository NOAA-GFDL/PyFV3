from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl.dsl.typing import FloatField


def rescale_delz_1(
    delz: FloatField,
    delp: FloatField,
):
    with computation(PARALLEL), interval(...):
        delz = -delz / delp


def rescale_delz_2(
    delz: FloatField,
    dp: FloatField,
):
    with computation(PARALLEL), interval(...):
        delz = -delz * dp
