from ndsl.constants import GRAV, RADIUS, RDGAS
from ndsl.dsl.gt4py import BACKWARD, FORWARD, computation
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ


@gtfunction
def average_gravity(grav_var: FloatField, grav_var_h: FloatField):
    return 0.5 * (grav_var_h[0, 0, 0] + grav_var_h[0, 0, 1])


# May not need this stencil at all
def average_gravity_stencil_defn(grav_var: FloatField, grav_var_h: FloatField):
    """
    Args:
        grav_var (out): gravity field
        grav_var_h (in): gravity value at interfaces
    """
    with computation(FORWARD), interval(...):
        grav_var = average_gravity(grav_var, grav_var_h)


def adjust_gravity(
    grav_var: FloatField,
    grav_var_h: FloatField,
    phis: FloatFieldIJ,
    delz: FloatField,
):
    """
    Args:
        grav_var (out): gravity field
        grav_var_h (out): gravity value at interfaces
        phis (in): geopotential
        delz (in): change in vertical height
    """
    with computation(FORWARD), interval(-1, None):
        newrad = RADIUS + (phis / GRAV)
        grav_var_h = GRAV * (RADIUS**2) / newrad**2

    with computation(BACKWARD), interval(0, -1):
        newrad = RADIUS + (phis / GRAV) - delz
        grav_var_h = GRAV * (RADIUS**2) / newrad**2
        grav_var = average_gravity(grav_var, grav_var_h)


def neg_rdgas_div_gravity(rdg: FloatField, grav_var: FloatField):
    """
    Adjust rdg to be the negative RDGAS divided by the variable gravity
    for Whole Atmosphere Modeling

    Args:
        rdg (out): negative radiative gas divided by variable gravity
        grav_var (in): variable gravity
    """
    with computation(FORWARD), interval(...):
        rdg = -RDGAS / grav_var
