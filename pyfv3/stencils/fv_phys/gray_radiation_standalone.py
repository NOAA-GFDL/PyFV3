from pyfv3.stencils.fv_phys.gray_radiation import GrayRadiation
from pyfv3.stencils.fv_phys._config import GrayRadiationConfig
import ndsl.constants as constants
from ndsl import Quantity, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, cos, interval, min, sin
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import GridData


def gather_inputs(
        delp: FloatField,
        peln: FloatField,
        delz: FloatField,
        p3: FloatField,
        den: FloatField,
):
    with computation(BACKWARD), interval(0, -1):
        p3 = delp / (peln[0, 0, 1] - peln[0, 0, 0])
        den = -delp/(constants.GRAV * delz)

@gtfunction
def save_low_clouds(
    ql: FloatField,
    qi: FloatField,
    qa: FloatField,
    clouds: FloatFieldIJ,
):
    if ((ql[0, 0, 0] > 1.e-5 or qi[0, 0, 0] > 2.e-4) and qa[0, 0, 0] > 1.e-3):
        return max(clouds, qa[0, 0, 0])
    else:
        return clouds

def get_low_clouds(
    ql: FloatField,
    qi: FloatField,
    qa: FloatField,
    clouds: FloatFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            clouds = 0.0
            clouds = save_low_clouds(ql, qi, qa, clouds)
        with interval(1, -1):
            clouds = save_low_clouds(ql, qi, qa, clouds)
        with interval(-1, None):
            clouds = save_low_clouds(ql, qi, qa, clouds)
            clouds = min(1.0, clouds)

def set_clouds(clouds: FloatFieldIJ):
    from __externals__ import low_cf0
    with computation(FORWARD), interval(0, 1):
        clouds = low_cf0

def maybe_calc_strat_rad_and_set_temp(
        p3: FloatField,
        t_dt: FloatField,
        t_dt_rad: FloatField,
        pt: FloatField,
        t_out: FloatField,
):
    from __externals__ import dt_atmos, heating, strat_rad
    with computation(PARALLEL), interval(...):
        t_dt = t_dt + t_dt_rad
        if strat_rad:
            if p3 < 100.e2:
                t_dt = t_dt + heating * (100.E2 - p3) / 100.E2

        t_out = pt + dt_atmos * t_dt


class GrayRadSolo:
    """
    Wrapper to run the gray radiation scheme as a standalone parameterization
    instead of as part of the fv physics package
    """
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        config: GrayRadiationConfig,
    ):
        self._prog_low_cloud = config.prog_low_cloud

        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [I_DIM, J_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity() -> Quantity:
            return quantity_factory.zeros(
                [I_DIM, J_DIM, K_DIM],
                units="unknown",
                dtype=Float,
            )

        self._clouds = make_quantity_2d()
        self._t_dt_rad = make_quantity()
        self._p3 = make_quantity()
        self._den = make_quantity()

        self._gather_inputs = stencil_factory.from_dims_halo(
            gather_inputs,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
        )

        if self._prog_low_cloud:
            origin = stencil_factory.grid_indexing.origin
            domain = stencil_factory.grid_indexing.domain
            k2 = int(domain[2]/2)
            dk2 = int(domain[2] - k2)
            self._get_low_clouds = stencil_factory.from_origin_domain(
                get_low_clouds,
                origin=(origin[0], origin[1], k2),
                domain=(domain[0], domain[1], dk2),
            )
        else:
            self._set_clouds = stencil_factory.from_dims_halo(
                set_clouds,
                compute_dims=[I_DIM, J_DIM],
                externals={
                    "low_cf0": config.low_cf0,
                },
            )

        self._gray_rad = GrayRadiation(
            stencil_factory,
            quantity_factory,
            grid_data,
            config
        )

        heating = config.heating_rate / 86400.
        self._maybe_calc_strat_rad_and_set_temp = stencil_factory.from_dims_halo(
            maybe_calc_strat_rad_and_set_temp,
            compute_dims=[I_DIM, J_DIM, K_DIM],
            externals={
                "dt_atmos": config.dt_atmos,
                "heating": heating,
                "strat_rad": config.strat_rad,
            },
        )
        
    def __call__(
        self,
        pt,
        ql,
        qi,
        qa,
        delp,
        pe,
        peln,
        ps,
        delz,
        t_dt,
        ts,
        olr,
        lwu,
        lwd,
        sw_surf,
        t_out,
    ):
        self._gather_inputs(
            delp,
            peln,
            delz,
            self._p3,
            self._den,
        )

        if self._prog_low_cloud:
            self._get_low_clouds(
                ql,
                qi,
                qa,
                self._clouds,
            )
        else:
            self._set_clouds(
                self._clouds,
            )

        self._gray_rad(
            self._clouds,
            ts,
            pt,
            ps,
            pe,
            delz,
            self._den,
            self._t_dt_rad,
            olr,
            lwu,
            lwd,
            sw_surf,
        )

        self._maybe_calc_strat_rad_and_set_temp(
            self._p3,
            t_dt,
            self._t_dt_rad,
            pt,
            t_out,
        )
