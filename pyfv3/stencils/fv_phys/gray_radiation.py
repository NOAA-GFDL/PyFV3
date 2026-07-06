from ndsl import Quantity, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, I_INTERFACE_DIM, J_DIM, J_INTERFACE_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.grid import GridData


def gray_radiation():
    with computation(PARALLEL), interval(...):
        pass


class GrayRadiation:
    """
    Gray-Radiation algorithms based on Frierson, Held, and Zurita-Gotor, 2006 JAS
    Note: delz is negative
    Fortran implementation Coded by S.-J. Lin, June 20, 2012
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        dt_atmos: float
    ):
        self._lon = grid_data._horizontal_data.lon
        self._lat = grid_data._horizontal_data.lat
        pass

    def __call__(
        self,
        clouds: FloatFieldIJ,
        ts: FloatFieldIJ,
        temperature: FloatField,
        ps: FloatFieldIJ,
        phalf: FloatField,
        delz: FloatField,
        rho: FloatField,
        t_dt: FloatField,
        olr: FloatFieldIJ,
        lwu: FloatFieldIJ,
        lwd: FloatFieldIJ,
        sw_surf: FloatFieldIJ,
    ):
        pass