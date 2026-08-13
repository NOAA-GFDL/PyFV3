import dace
import numpy as np

from ndsl import (
    NDSLRuntime,
    QuantityFactory,
    StencilFactory,
    SubtileGridSizer,
    constants,
)
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    K_DIM,
    SECONDS_PER_DAY,
)
from ndsl.dsl.gt4py import __INLINED, BACKWARD, FORWARD, PARALLEL, computation, float64
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import horizontal, interval, log, region, sin
from ndsl.dsl.typing import Float, FloatField, FloatFieldK

SDAY = 86400.0


# NOTE: The fortran version of this computes rf in the first timestep only. Then
# rf_initialized let's you know you can skip it. Here we calculate it every
# time.
@gtfunction
def compute_rf_vals(pfull, bdt, rf_cutoff, tau0, ptop):
    return (
        bdt
        / tau0
        * sin(0.5 * constants.PI * log(rf_cutoff / pfull) / log(rf_cutoff / ptop)) ** 2
    )


@gtfunction
def compute_rff_vals(pfull, dt, rf_cutoff, tau0, ptop):
    rffvals = compute_rf_vals(pfull, dt, rf_cutoff, tau0, ptop)
    rffvals = float64(1.0) / (float64(1.0) + rffvals)
    return rffvals


@gtfunction
def dm_layer(rf, dp, wind):
    return (1.0 - rf) * dp * wind


def ray_fast_damping_increment(
    pfull: FloatFieldK,
    dt: Float,
    ptop: Float,
    rf: FloatField,
):
    """rf is rayleigh damping increment, fraction of vertical velocity
    left after doing rayleigh damping (w -> w * rf)
    """
    from __externals__ import rf_cutoff, tau

    with computation(PARALLEL), interval(...):
        if pfull < rf_cutoff:
            # rf is rayleigh damping increment, fraction of vertical velocity
            # left after doing rayleigh damping (w -> w * rf)
            rf = compute_rff_vals(pfull, dt, rf_cutoff, tau * SECONDS_PER_DAY, ptop)


def ray_fast_wind_compute(
    u: FloatField,
    v: FloatField,
    w: FloatField,
    delta_p_ref: FloatFieldK,  # reference delta pressure
    pfull: FloatFieldK,  # input layer pressure reference?
    rf: FloatFieldK,
    rf_cutoff_nudge: Float,
):
    """
    Args:
        u (inout):
        v (inout):
        w (inout):
        delta_p_ref (in):
        pfull (in):
        dt (in):
        ptop (in):
        rf_cutoff_nudge (in):
        ks (in):
    """
    from __externals__ import hydrostatic, local_ie, local_je, rf_cutoff

    # dm_stencil
    with computation(FORWARD):
        with interval(0, 1):
            if pfull < rf_cutoff_nudge:
                p_ref = delta_p_ref
        with interval(1, None):
            p_ref = p_ref[0, 0, -1]
            if pfull < rf_cutoff_nudge:
                p_ref += delta_p_ref
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff_nudge:
            p_ref = p_ref[0, 0, 1]
    # ray_fast_wind(u)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[: local_ie + 1, :]):
                if pfull < rf_cutoff:
                    # dmdir = (1.0 - rf) * dp * wind
                    dmdir = dm_layer(rf, delta_p_ref, u)
                    u *= rf
                else:
                    p_ref = 0
        with interval(1, None):
            with horizontal(region[: local_ie + 1, :]):
                dmdir = dmdir[0, 0, -1]
                if pfull < rf_cutoff:
                    dmdir += dm_layer(rf, delta_p_ref, u)
                    u *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, :]):
            if pfull < rf_cutoff_nudge:
                u += dmdir / p_ref
    # ray_fast_wind(v)
    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(region[:, : local_je + 1]):
                if pfull < rf_cutoff:
                    dmdir = dm_layer(rf, delta_p_ref, v)
                    v *= rf
                else:
                    p_ref = 0
        with interval(1, None):
            with horizontal(region[:, : local_je + 1]):
                dmdir = dmdir[0, 0, -1]
                if pfull < rf_cutoff:
                    dmdir += dm_layer(rf, delta_p_ref, v)
                    v *= rf
    with computation(BACKWARD), interval(0, -1):
        if pfull < rf_cutoff:
            dmdir = dmdir[0, 0, 1]
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, : local_je + 1]):
            if pfull < rf_cutoff_nudge:
                v += dmdir / p_ref
    # ray_fast_w
    with computation(PARALLEL), interval(...):
        with horizontal(region[: local_ie + 1, : local_je + 1]):
            if __INLINED(not hydrostatic):
                if pfull < rf_cutoff:
                    w *= rf


class RayleighDamping(NDSLRuntime):
    """
    Apply Rayleigh damping (for tau > 0).

    Namelist:
        - tau [Float]: time scale (in days) for Rayleigh friction applied to horizontal
                       and vertical winds; lost kinetic energy is converted to heat,
                       except on nested grids.
        - rf_cutoff [Float]: pressure below which no Rayleigh damping is applied
                             if tau > 0.

    Fortran name: ray_fast.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        rf_cutoff: Float,
        tau: Float,
        hydrostatic: bool,
    ):
        super().__init__(stencil_factory)

        grid_indexing = stencil_factory.grid_indexing
        self._rf_cutoff = Float(rf_cutoff)
        origin, domain = grid_indexing.get_origin_domain(
            [I_INTERFACE_DIM, J_INTERFACE_DIM, K_DIM]
        )

        if tau == 0:
            raise NotImplementedError(
                "Dynamical Core (fv_dynamics): RayleighDamping, with tau <= 0, is not implemented"
            )

        ax_offsets = grid_indexing.axis_offsets(origin, domain)
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._ray_fast_wind_compute = stencil_factory.from_origin_domain(
            ray_fast_wind_compute,
            origin=origin,
            domain=domain,
            externals={
                "hydrostatic": hydrostatic,
                "rf_cutoff": self._rf_cutoff,
                "tau": tau,
                **local_axis_offsets,
            },
        )

        self._ray_fast_damping_increment = stencil_factory.from_origin_domain(
            ray_fast_damping_increment,
            origin=(0, 0, origin[2]),
            domain=(1, 1, domain[2]),
            externals={
                "rf_cutoff": self._rf_cutoff,
                "tau": tau,
            },
        )
        sizer = SubtileGridSizer(
            nx=1,
            ny=1,
            nz=domain[2],
            n_halo=0,
            data_dimensions={},
            backend=stencil_factory.backend,
        )

        # Not a local - because of the separate factory trick
        K_quantity_factory = QuantityFactory(sizer, backend=stencil_factory.backend)
        self._tmp_damping_increment = K_quantity_factory.ones(
            [I_DIM, J_DIM, K_DIM], "n/a"
        )

        # Not a local because it's a lazy initialization
        self._damping_increment = quantity_factory.ones([K_DIM], "")

        self._initialize_damping_increment = np.ones((1,), dtype=bool)
        self._KM = domain[2]

    def __call__(
        self,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        dp: FloatFieldK,
        pfull: FloatFieldK,
        dt: Float,
        ptop: Float,
    ):
        """
        Args:
            u (inout)
            v (inout)
            w (inout)
            dp (in)
            pfull (in)
            dt (in)
            ptop (in)
        """
        rf_cutoff_nudge = self._rf_cutoff + min(Float(100.0), Float(10.0) * ptop)

        # TODO: this is a bad fix to go around an orchestration issue
        #       on compile-time values. Do better.
        if self._initialize_damping_increment[0]:
            self._ray_fast_damping_increment(
                pfull=pfull,
                dt=dt,
                ptop=ptop,
                rf=self._tmp_damping_increment,
            )
            for _k in dace.map[0 : self._KM]:
                self._damping_increment[_k] = self._tmp_damping_increment[0, 0, _k]
            self._initialize_damping_increment[0] = False
        self._ray_fast_wind_compute(
            u=u,
            v=v,
            w=w,
            delta_p_ref=dp,
            pfull=pfull,
            rf=self._damping_increment,
            rf_cutoff_nudge=rf_cutoff_nudge,
        )
