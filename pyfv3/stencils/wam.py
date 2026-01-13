from ndsl.constants import GRAV, RADIUS, RDGAS, X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, computation
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl import StencilFactory


@gtfunction
def average_gravity(grav_var: FloatField, grav_var_h: FloatField):
    grav_var = 0.5 * (grav_var_h[0, 0, 0] + grav_var_h[0, 0, 1])
    return grav_var


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
        newrad = RADIUS + (phis / GRAV)
        newrad = newrad - delz
        grav_var_h = GRAV * (RADIUS**2) / newrad**2
        grav_var = average_gravity(grav_var, grav_var_h)

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
        rdg = - RDGAS / grav_var

class WholeAtmos:
    def __init__(self, stencil_factory: StencilFactory):
        self.constructed_average_gravity_stencil = stencil_factory.from_dims_halo(
            func=average_gravity_stencil_defn,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.constructed_adjust_gravity_stencil = stencil_factory.from_dims_halo(
            func=adjust_gravity,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self.constructed_neg_rdgas_div_gravity_stencil = stencil_factory.from_dims_halo(
            func=neg_rdgas_div_gravity,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, grav_var: FloatField, grav_var_h: FloatField, rdg: FloatField, phis: FloatFieldIJ, delz: FloatField):
        self.constructed_adjust_gravity_stencil(grav_var, grav_var_h, phis, delz)
        self.constructed_neg_rdgas_div_gravity_stencil(rdg, grav_var)