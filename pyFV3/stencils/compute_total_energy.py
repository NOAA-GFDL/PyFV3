from ndsl import StencilFactory, QuantityFactory
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM, GRAV
from pyFV3._config import DynamicalCoreConfig
from pyFV3.tracers import Tracers
from pyFV3.stencils.moist_cv import moist_cv_nwat6_fn
from gt4py.cartesian.gtscript import (
    BACKWARD,
    FORWARD,
    interval,
    computation,
    K,
)
from ndsl.grid import GridData


def _compute_total_energy__stencil(
    hs: FloatFieldIJ,  # type: ignore
    delp: FloatField,  # type: ignore
    delz: FloatField,  # type: ignore
    qc: FloatField,  # type:ignore
    pt: FloatField,  # type: ignore
    u: FloatField,  # type: ignore
    v: FloatField,  # type: ignore
    w: FloatField,  # type: ignore
    qvapor: FloatField,  # type: ignore
    qliquid: FloatField,  # type: ignore
    qrain: FloatField,  # type: ignore
    qsnow: FloatField,  # type: ignore
    qice: FloatField,  # type: ignore
    qgraupel: FloatField,  # type: ignore
    rsin2: FloatFieldIJ,  # type: ignore
    cosa_s: FloatFieldIJ,  # type: ignore
    phyz: FloatField,  # type: ignore
    te_2d: FloatFieldIJ,  # type: ignore
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
        qvapor(in):
        qliquid(in):
        qrain(in):
        qsnow(in):
        qice(in):
        qgraupel(in):
        rsin2(in):
        cosa_s(in):
        phyz(inout):
        te_2d(out):
    """

    with computation(BACKWARD), interval(-1, None):
        te_2d = 0.0
        phis = hs
    with computation(BACKWARD), interval(0, -1):
        phis = phis[K + 1] - GRAV * delz
    with computation(FORWARD), interval(0, -1):
        cvm, qd = moist_cv_nwat6_fn(qvapor, qliquid, qrain, qsnow, qice, qgraupel)

        te_2d = te_2d + delp * (
            cvm * pt * (1.0 + qc) * (1.0 - qd)
            + 0.5
            * (
                phis
                + phis[0, 0, 1]
                + w**2.0
                + 0.5
                * rsin2
                * (
                    u**2.0
                    + u[0, 1, 0] ** 2.0
                    + v**2.0
                    + v[1, 0, 0] ** 2.0
                    - (u + u[0, 1, 0]) * (v + v[1, 0, 0]) * cosa_s
                )
            )
        )


class ComputeTotalEnergy:
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
        if config.hydrostatic:
            raise NotImplementedError(
                "Dynamics (Compute Total Energy): "
                " hydrostatic option is not implemented."
            )

        if not config.moist_phys:
            raise NotImplementedError(
                "Dynamics (Compute Total Energy): "
                " moist_phys=False option is not implemented."
            )

        self._phyz = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Unknown",
            dtype=Float,
        )

        self._compute_total_energy = stencil_factory.from_dims_halo(
            func=_compute_total_energy__stencil,
            compute_dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
        )
        self._rsin2 = grid_data.rsin2
        self._cosa_s = grid_data.cosa_s

    def __call__(
        self,
        hs: FloatFieldIJ,  # type: ignore
        delp: FloatField,  # type: ignore
        delz: FloatField,  # type: ignore
        qc: FloatField,  # type:ignore
        pt: FloatField,  # type: ignore
        u: FloatField,  # type: ignore
        v: FloatField,  # type: ignore
        w: FloatField,  # type: ignore
        tracers: Tracers,
        te_2d: FloatFieldIJ,  # type: ignore
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
            qvapor=tracers["vapor"],
            qliquid=tracers["liquid"],
            qrain=tracers["rain"],
            qsnow=tracers["snow"],
            qice=tracers["ice"],
            qgraupel=tracers["graupel"],
            rsin2=self._rsin2,
            cosa_s=self._cosa_s,
            phyz=self._phyz,
            te_2d=te_2d,
        )
