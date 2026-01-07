import ndsl.constants as constants
from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.typing import FloatField


def neg_rdgas_div_gravity(rdg: FloatField, grav_var: FloatField):
    """
    # JK TODO: Is there a better name than this?
    Adjust rdg to be the negative RDGAS divided by the variable gravity
    for Whole Atmosphere Modeling

    Args:
        rdg (out): negative radiative gas divided by variable gravity
        grav_var (in): variable gravity
    """
    with computation(FORWARD), interval(...):
        rdg = - constants.RDGAS / grav_var
