import ndsl.constants as constants
from ndsl import Quantity, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, cos, interval, max, sin
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import GridData
from pyfv3.stencils.fv_phys._config import GrayRadiationConfig


T0E =  8.  # Dargan value= 6.
T0P = 1.5
FL = 0.1
SBC = 5.6734E-8
CP = 1004.64
SOLAR_CONSTANT = 1367.
ALBD = 0.0

@gtfunction
def compute_tau(
    phalf: FloatField,
    ps: FloatFieldIJ,
    tau0: FloatFieldIJ
):
    from __externals__ import strat_off
    if not strat_off:
        # Dargan version:
        sig = phalf[0, 0, 0] / ps[0, 0]
        tau = tau0[0, 0] * (sig * FL + (1. - FL) * sig**4)
    else:
        # SJL: less cooling for the stratosphere
        tau = tau0[0, 0] * (phalf[0, 0, 0] / ps[0, 0])**4
    return tau

def gray_radiation(
    lon: FloatFieldIJ,
    lat: FloatFieldIJ,
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
    tau0: FloatFieldIJ,
    solar_ang: FloatFieldIJ,
):
    from __externals__ import dt_atmos, diurnal_cycle, sw_rad, shift_n
    with computation(FORWARD), interval(0, 1):
        tau0 = T0E + (T0P - T0E) * (sin(lat) * sin(lat))
        # Annual & global mean solar_abs ~ 0.25 * 1367 * ( 1-0.3) ~ 240
        # Earth cross section/total_area = 0.25; 0.3 = Net cloud reflection and atm abs
        if diurnal_cycle:
            solar_ang[0, 0] = 2 * constants.pi * dt_atmos / 86400. + lon[0, 0]
            sw_surf[0, 0] = sw_rad * (1. - clouds[0, 0]) * cos(lat[0, 0]) * max(
                0., cos(solar_ang[0, 0])
            )
            sw_surf[0, 0] = sw_surf[0, 0] * (1. - ALBD)
        else:
            sw_surf = sw_rad * (1. - clouds) * max(
                0., cos(lat - shift_n * (constants.pi / 180.))
            ) * (1.-ALBD)
    
    with computation(BACKWARD):
        with interval(-1, None):
            tau = compute_tau(phalf, ps, tau0)
        with interval(0, -1):
            tau = compute_tau(phalf, ps, tau0)
            delt = tau[0, 0, 1] - tau[0, 0, 0]
            b = SBC * temperature**4
    
    # top down integration:
    with computation(FORWARD):
        with interval(0, 1):
            dr = 0.0
        with interval(1, None):
            dr = (dr[0, 0, -1]+delt[0, 0, -1]*(b[0, 0, -1]-0.5*dr[0, 0, -1]))/(1.+0.5*delt[0, 0, -1])

    # Bottom up:
    with computation(BACKWARD):
        with interval(-1, None):
            ur = SBC*ts[0, 0]**4
            lwu(i) = ur
            lw = ur - dr
        with interval(1, -1):
            ur = (ur[0, 0, 1]+delt[0, 0, 0]*(b[0, 0, 0]-0.5*ur[0, 0, 1]))/(1.+0.5*delt[0, 0, 0])
            lw = ur - dr

    # Compute net long wave cooling rate:
    with computation(PARALLEL), interval(...):
        t_dt[0, 0, 0] = (lw[0, 0, 0] - lw[0, 0, 1])/(CP*rho[0, 0, 0]*delz[0, 0, 0])
    with computation(PARALLEL):
        with interval(0, 1):
            olr[0, 0] = ur[0, 0, 0]
        with interval(-1, None):
            lwd[0, 0] = dr[0, 0, 0]


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
        config: GrayRadiationConfig,
    ):
        self._lon = grid_data._horizontal_data.lon
        self._lat = grid_data._horizontal_data.lat
        if config.diurnal_cycle:
            sw_rad = SOLAR_CONSTANT * (1. - config.sw_abs)
        else:
            sw_rad = (1./constants.pi) * SOLAR_CONSTANT*(1. - config.sw_abs)
        
        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [I_DIM, J_DIM],
                units="unknown",
                dtype=Float,
            )
        
        self._tau0 = make_quantity_2d()
        self._solar_ang = make_quantity_2d()

        self._gray_radiation = stencil_factory.from_dims_halo(
            gray_radiation,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            externals={
                "dt_atmos": config.dt_atmos,
                "diurnal_cycle": config.diurnal_cycle,
                "sw_rad": sw_rad,
                "shift_n": config.shift_n,
                "strat_off": config.strat_off
            },
        )

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
        self._gray_radiation(
            self._lon,
            self._lat,
            clouds,
            ts,
            temperature,
            ps,
            phalf,
            delz,
            rho,
            t_dt,
            olr,
            lwu,
            lwd,
            sw_surf,
            self._tau0,
            self._solar_ang,
        )
