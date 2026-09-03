from gt4py.cartesian.gtscript import (  # isort: skip
    __INLINED,
    BACKWARD,
    PARALLEL,
    K,
    computation,
    interval,
    FORWARD,
)

from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import GRAV, I_DIM, J_DIM, K_INTERFACE_DIM
from ndsl.dsl.typing import FloatField, FloatFieldIJ
from ndsl.grid import GridData
from pyfv3._config import DynamicalCoreConfig
from pyfv3.stencils.moist_cv import moist_cv_nwat0_fn, moist_cv_nwat6_fn
from pyfv3.tracers import FVTracers


def _compute_total_energy__stencil(
    hs: FloatFieldIJ,
    delp: FloatField,
    delz: FloatField,
    qc: FloatField,
    pt: FloatField,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    tracers: FVTracers,
    rsin2: FloatFieldIJ,
    cosa_s: FloatFieldIJ,
    te_2d: FloatFieldIJ,
):
    """
    Dev Note: this is _very_ close to moist_cv.moist_te. The only numerical differences
    is that the te/te_2d computation as an extra (1.+qc(i,j,k))*(1.-qd(i))

    Args:
        hs(in):
        delp(in):
        delz(in):
        pt(in):
        qc(in):
        u(in):
        v(in):
        w(in):
        tracers(in):
        rsin2(in):
        cosa_s(in):
        te_2d(out):
    """

    from __externals__ import i_graupel, i_ice, i_liquid, i_rain, i_snow, i_vapor, nwat

    with computation(BACKWARD), interval(-1, None):
        te_2d = 0.0
        phis = hs
    with computation(BACKWARD), interval(0, -1):
        phis = phis[K + 1] - GRAV * delz
    with computation(PARALLEL), interval(0, -1):
        if __INLINED(nwat == 0):
            cvm, qd = moist_cv_nwat0_fn()
        elif __INLINED(nwat == 6):
            cvm, qd = moist_cv_nwat6_fn(
                tracers.A[i_vapor],
                tracers.A[i_liquid],
                tracers.A[i_rain],
                tracers.A[i_snow],
                tracers.A[i_ice],
                tracers.A[i_graupel],
            )
    with computation(FORWARD), interval(0, -1):
        te_2d = te_2d + delp * (
            cvm * pt * (1.0 + qc) * (1.0 - qd)
            + 0.5
            * (
                phis
                + phis[0, 0, 1]
                + w**2
                + 0.5
                * rsin2
                * (
                    u**2
                    + u[0, 1, 0] ** 2
                    + v**2
                    + v[1, 0, 0] ** 2
                    - (u + u[0, 1, 0]) * (v + v[1, 0, 0]) * cosa_s
                )
            )
        )


class ComputeTotalEnergy(NDSLRuntime):
    """Compute total energy performs the FV3-consistent
    computation of the global total energy.

    It includes the potential, internal (latent and sensible heat), kinetic terms."""

    def __init__(
        self,
        config: DynamicalCoreConfig,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
    ) -> None:
        super().__init__(stencil_factory)

        if config.hydrostatic:
            raise NotImplementedError(
                "Dynamics (Compute Total Energy): hydrostatic option is not implemented."
            )

        if not config.moist_phys:
            raise NotImplementedError(
                "Dynamics (Compute Total Energy): moist_phys=False option is not implemented."
            )

        if config.nwat not in [0, 6]:
            raise NotImplementedError(
                f"Compute total energy not implemented for {config.nwat} water species."
            )

        self._compute_total_energy = stencil_factory.from_dims_halo(
            func=_compute_total_energy__stencil,
            compute_dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            externals={
                "nwat": config.nwat,
                "i_vapor": FVTracers.index("vapor"),
                "i_liquid": FVTracers.index("liquid") if config.nwat == 6 else -1,
                "i_rain": FVTracers.index("rain") if config.nwat == 6 else -1,
                "i_ice": FVTracers.index("ice") if config.nwat == 6 else -1,
                "i_snow": FVTracers.index("snow") if config.nwat == 6 else -1,
                "i_graupel": FVTracers.index("graupel") if config.nwat == 6 else -1,
            },
        )
        self._rsin2 = grid_data.rsin2
        self._cosa_s = grid_data.cosa_s

    def __call__(
        self,
        hs: FloatFieldIJ,
        delp: FloatField,
        delz: FloatField,
        qc: FloatField,
        pt: FloatField,
        u: FloatField,
        v: FloatField,
        w: FloatField,
        tracers: FVTracers,
        te_2d: FloatFieldIJ,
    ) -> None:
        self._compute_total_energy(
            hs=hs,
            delp=delp,
            delz=delz,
            qc=qc,
            pt=pt,
            u=u,
            v=v,
            w=w,
            tracers=tracers,
            rsin2=self._rsin2,
            cosa_s=self._cosa_s,
            te_2d=te_2d,
        )
