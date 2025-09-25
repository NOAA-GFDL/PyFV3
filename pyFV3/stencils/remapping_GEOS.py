from gt4py.cartesian.gtscript import FORWARD, computation, interval

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.comm.communicator import Communicator
from ndsl.constants import (
    CV_AIR,
    GRAV,
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, FloatFieldIJ64, FloatFieldK
from ndsl.grid import GridData
from ndsl.stencils.basic_operations import adjust_divide_stencil
from pyFV3._config import RemappingConfig
from pyFV3.mpi.sum import GlobalSum
from pyFV3.stencils import moist_cv
from pyFV3.stencils.map_single import MapSingle
from pyFV3.stencils.mapn_tracer import MapNTracer
from pyFV3.stencils.moist_cv import moist_pt_last_step
from pyFV3.stencils.remapping import (
    CONSV_MIN,
    init_pe,
    moist_cv_pt_pressure,
    pe0_ptop_xmax,
    pe_pk_delp_peln,
    pn2_pk_delp,
    pressures_mapu,
    pressures_mapv,
)
from pyFV3.stencils.saturation_adjustment import SatAdjust3d
from pyFV3.stencils.scale_delz import rescale_delz_1, rescale_delz_2
from pyFV3.stencils.w_fix_consrv_moment import W_fix_consrv_moment
from pyFV3.tracers import TracersType


def _normalize_to_grid_stencil(
    te_2d: FloatFieldIJ, zsum_2d: FloatFieldIJ, area: FloatFieldIJ64
):
    with computation(FORWARD), interval(0, 1):
        te_2d = te_2d * area
        zsum_2d = zsum_2d * area


class LagrangianToEulerian_GEOS:
    """
    GEOS v11.4.2 remapping - derived from original fvcore.

    Fortran name is Lagrangian_to_Eulerian
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: RemappingConfig,
        comm: Communicator,
        grid_data: GridData,
        nq,
        pfull,
        tracers: TracersType,
        adiabatic: bool,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        self._comm = comm
        self._stencil_factory = stencil_factory
        grid_indexing = stencil_factory.grid_indexing

        # Configuration
        hydrostatic = config.hydrostatic
        if hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")

        if adiabatic:
            raise NotImplementedError("Adiabatic is not implemented")

        self._t_min = Float(184.0)
        self._nq = nq
        self._w_max = Float(90.0)
        self._w_min = Float(-60.0)
        self._area_64 = grid_data.area_64
        self._cosa_s = grid_data.cosa_s
        self._rsin2 = grid_data.rsin2
        self._kord_tm = abs(config.kord_tm)
        self._kord_wz = config.kord_wz
        self._kord_mt = config.kord_mt
        self._do_sat_adjust = config.do_sat_adj
        self._adiabatic = adiabatic
        self.kmp = grid_indexing.domain[2] - 1
        for k in range(pfull.shape[0]):
            if pfull.view[k] > 10.0e2:
                self.kmp = k
                break
        # do_omega = hydrostatic and last_step # TODO pull into inputs

        # Quantities
        self._pe1 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pe2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pe3 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="Pa",
            dtype=Float,
        )
        self._dp2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pn1 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pn2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pe0 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="Pa",
            dtype=Float,
        )
        self._pe3 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="Pa",
            dtype=Float,
        )

        self._gz = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="m^2 s^-2",
            dtype=Float,
        )
        self._cvm = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._compute_performed = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="mask",
            dtype=bool,
        )
        self._w2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="temp W",
            dtype=Float,
        )
        self._pk2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Pa",
            dtype=Float,
        )

        self._te_2d = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="Pa",
            dtype=Float,
        )

        self._zsum1 = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="Pa",
            dtype=Float,
        )
        self._phis = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            units="n/a",
            dtype=Float,
        )

        # Stencils

        self._global_sum = GlobalSum(
            communicator=comm,
            quantity_factory=quantity_factory,
            grid_indexing=stencil_factory.grid_indexing,
        )

        self._init_pe = stencil_factory.from_origin_domain(
            init_pe,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 1, 1)),
        )

        self._moist_cv_pt_pressure = stencil_factory.from_origin_domain(
            moist_cv_pt_pressure,
            externals={"hydrostatic": hydrostatic},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )

        self._pn2_pk_delp = stencil_factory.from_origin_domain(
            pn2_pk_delp,
            origin=grid_indexing.origin_compute(add=(0, 0, 1)),
            domain=grid_indexing.domain_compute(add=(0, 0, -1)),
        )

        self._map_single_pt = MapSingle(
            stencil_factory,
            quantity_factory,
            self._kord_tm,
            mode=1,
            dims=[X_DIM, Y_DIM, Z_DIM],
            interpolate_contribution=True,
        )

        self._mapn_tracer = MapNTracer(
            stencil_factory,
            quantity_factory,
            kord=abs(config.kord_tr),
            fill=config.fill,
            tracers=tracers,
        )

        self._map_single_w = MapSingle(
            stencil_factory,
            quantity_factory,
            self._kord_wz,
            mode=-2,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._map_single_delz = MapSingle(
            stencil_factory,
            quantity_factory,
            self._kord_wz,
            mode=1,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._moist_cv_pkz = stencil_factory.from_origin_domain(
            moist_cv.moist_pkz,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._pressures_mapu = stencil_factory.from_origin_domain(
            pressures_mapu,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 1, 1)),
        )

        self._map_single_u = MapSingle(
            stencil_factory,
            quantity_factory,
            self._kord_mt,
            mode=-1,
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
        )

        self._pressures_mapv = stencil_factory.from_origin_domain(
            pressures_mapv,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(1, 0, 1)),
        )

        self._map_single_v = MapSingle(
            stencil_factory,
            quantity_factory,
            self._kord_mt,
            mode=-1,
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
        )

        self._saturation_adjustment = SatAdjust3d(
            stencil_factory, config.sat_adjust, self._area_64, self.kmp
        )

        self._moist_cv_last_step_stencil = stencil_factory.from_origin_domain(
            moist_pt_last_step,
            origin=(grid_indexing.isc, grid_indexing.jsc, 0),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),
        )

        self._fill_cond = stencil_factory.from_origin_domain(
            moist_cv.cond_output,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._adjust_divide = stencil_factory.from_origin_domain(
            adjust_divide_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._rescale_delz_1 = stencil_factory.from_origin_domain(
            rescale_delz_1,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._rescale_delz_2 = stencil_factory.from_origin_domain(
            rescale_delz_2,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._w_fix_consrv_moment = stencil_factory.from_origin_domain(
            func=W_fix_consrv_moment,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._pe0_ptop_xmax = stencil_factory.from_origin_domain(
            pe0_ptop_xmax,
            origin=(
                grid_indexing.n_halo + grid_indexing.domain[0],
                grid_indexing.n_halo,
                0,
            ),
            domain=(1, grid_indexing.domain[1] + 1, 1),
        )
        self._pe_pk_delp_peln = stencil_factory.from_origin_domain(
            pe_pk_delp_peln,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )
        self._moist_cv_te = stencil_factory.from_origin_domain(
            moist_cv.moist_te,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        )

        self._te_zsum = stencil_factory.from_origin_domain(
            moist_cv.te_zsum,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._normalize_to_grid = stencil_factory.from_origin_domain(
            _normalize_to_grid_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        tracers: TracersType,
        pt: FloatField,  # type: ignore
        delp: FloatField,  # type: ignore
        delz: FloatField,  # type: ignore
        peln: FloatField,  # type: ignore
        u: FloatField,  # type: ignore
        v: FloatField,  # type: ignore
        w: FloatField,  # type: ignore
        mfx: FloatField,  # type: ignore
        mfy: FloatField,  # type: ignore
        cx: FloatField,  # type: ignore
        cy: FloatField,  # type: ignore
        cappa: FloatField,  # type: ignore
        q_con: FloatField,  # type: ignore
        pkz: FloatField,  # type: ignore
        pk: FloatField,  # type: ignore
        pe: FloatField,  # type: ignore
        hs: FloatFieldIJ,  # type: ignore
        te0_2d: FloatFieldIJ,  # type: ignore
        ps: FloatFieldIJ,  # type: ignore
        wsd: FloatFieldIJ,  # type: ignore
        ak: FloatFieldK,  # type: ignore
        bk: FloatFieldK,  # type: ignore
        dp1: FloatField,  # type: ignore
        ptop: Float,  # type: ignore
        akap: Float,  # type: ignore
        zvir: Float,  # type: ignore
        last_step: bool,
        consv_te: Float,  # type: ignore
        mdt: Float,  # type: ignore
    ):
        """
        Remap the deformed Lagrangian surfaces onto the reference, or "Eulerian",
        coordinate levels.

        Args:
            tracers (inout): Tracer species tracked across
            pt (inout): D-grid potential temperature
            delp (inout): Pressure Thickness
            delz (in): Vertical thickness of atmosphere layers
            peln (inout): Logarithm of interface pressure
            u (inout): D-grid x-velocity
            v (inout): D-grid y-velocity
            w (inout): Vertical velocity
            ua (inout): A-grid x-velocity
            va (inout): A-grid y-velocity
            cappa (inout): Power to raise pressure to
            q_con (out): Total condensate mixing ratio
            pkz (in): Layer mean pressure raised to the power of Kappa
            pk (out): Interface pressure raised to power of kappa, final acoustic value
            pe (in): Pressure at layer edges
            hs (in): Surface geopotential
            te0_2d (inout): Atmosphere total energy in columns
            ps (out): Surface pressure
            wsd (in): Vertical velocity of the lowest level
            omga (unused): Vertical pressure velocity
            ak (in): Atmosphere hybrid a coordinate (Pa)
            bk (in): Atmosphere hybrid b coordinate (dimensionless)
            pfull (in): Pressure full levels
            dp1 (out): Pressure thickness before dyn_core (only written
                if do_sat_adjust=True)
            ptop (in): The pressure level at the top of atmosphere
            akap (in): Poisson constant (KAPPA)
            zvir (in): Constant (Rv/Rd-1)
            last_step (in): Flag for the last step of k-split remapping
            consv_te (in): If True, conserve total energy
            mdt (in) : Remap time step
            bdt (in): Timestep
        """
        # Global structure:
        #   pe1 is initial lagrangian edge pressures
        #   pe2 is final Eulerian edge pressures

        # Build remapping profiles
        self._init_pe(pe, self._pe1, self._pe2, ptop)
        self._moist_cv_pt_pressure(
            qvapor=tracers.vapor,
            qliquid=tracers.liquid,
            qrain=tracers.rain,
            qsnow=tracers.snow,
            qice=tracers.ice,
            qgraupel=tracers.graupel,
            q_con=q_con,
            pt=pt,
            cappa=cappa,
            delp=delp,
            delz=delz,
            pe=pe,
            pe2=self._pe2,
            ak=ak,
            bk=bk,
            dp2=self._dp2,
            ps=ps,
            pn1=self._pn1,
            pn2=self._pn2,
            peln=peln,
            remap_t=True,
            r_vir=zvir,
        )
        self._pn2_pk_delp(
            pe2=self._pe2,
            pn2=self._pn2,
            pk=self._pk2,
            akap=akap,
        )

        # Now that we have the pressure profiles, we can start remapping

        # Map pressure
        self._map_single_pt(
            pt,
            self._pn1,
            self._pn2,
            qmin=self._t_min,
        )

        # Map all tracers
        self._mapn_tracer(self._pe1, self._pe2, self._dp2, tracers)

        # Map vertical wind
        self._map_single_w(w, self._pe1, self._pe2, qs=wsd)
        self._rescale_delz_1(delz, delp)
        self._map_single_delz(delz, self._pe1, self._pe2)
        self._rescale_delz_2(delz, self._dp2)
        self._w_fix_consrv_moment(
            w=w,
            w2=self._w2,
            dp2=self._dp2,
            gz=self._gz,
            w_max=self._w_max,
            w_min=self._w_min,
            compute_performed=self._compute_performed,
        )

        # Map horizontal winds, fluxes and courant number
        self._pressures_mapu(pe, ak, bk, self._pe0, self._pe3, ptop)
        self._pe0_ptop_xmax(self._pe0, ptop)
        self._map_single_u(u, self._pe0, self._pe3)
        self._map_single_u(mfy, self._pe0, self._pe3)
        self._map_single_u(cy, self._pe0, self._pe3)

        self._pressures_mapv(pe, ak, bk, self._pe0, self._pe3)
        self._map_single_v(v, self._pe0, self._pe3)
        self._map_single_v(mfx, self._pe0, self._pe3)
        self._map_single_v(cx, self._pe0, self._pe3)

        self._pe_pk_delp_peln(
            pe=pe,
            pk=pk,
            delp=delp,
            peln=peln,
            pe2=self._pe2,
            pk2=self._pk2,
            pn2=self._pn2,
            ak=ak,
            bk=bk,
            akap=akap,
            ptop=ptop,
        )

        self._moist_cv_pkz(
            qvapor=tracers.vapor,
            qliquid=tracers.liquid,
            qrain=tracers.rain,
            qsnow=tracers.snow,
            qice=tracers.ice,
            qgraupel=tracers.graupel,
            pkz=pkz,
            pt=pt,
            cappa=cappa,
            delp=delp,
            delz=delz,
            r_vir=zvir,
        )

        dtmp = 0.0
        if last_step:
            if consv_te > CONSV_MIN:
                self._moist_cv_te(
                    qvapor=tracers.vapor,
                    qliquid=tracers.liquid,
                    qrain=tracers.rain,
                    qsnow=tracers.snow,
                    qice=tracers.ice,
                    qgraupel=tracers.graupel,
                    u=u,
                    v=v,
                    w=w,
                    te=self._te_2d,
                    pt=pt,
                    phis=self._phis,
                    delp=delp,
                    rsin2=self._rsin2,
                    cosa_s=self._cosa_s,
                    hs=hs,
                    delz=delz,
                    grav=GRAV,
                )

                self._te_zsum(
                    te_2d=self._te_2d,
                    te0_2d=te0_2d,
                    delp=delp,
                    pkz=pkz,
                    zsum1=self._zsum1,
                )

                # We can normalize to the same array because
                # they are properly reset in the above stencils
                self._normalize_to_grid(self._te_2d, self._zsum1, self._area_64)

                tesum: Float = self._global_sum(self._te_2d)
                zsum: Float = self._global_sum(self._zsum1)
                dtmp = tesum / (CV_AIR * zsum)

            elif consv_te < -CONSV_MIN:
                raise NotImplementedError(
                    "Unimplemented/untested case consv("
                    + str(consv_te)
                    + ")  < -CONSV_MIN("
                    + str(-CONSV_MIN)
                    + ")"
                )

        if self._do_sat_adjust:
            fast_mp_consv = consv_te > CONSV_MIN
            self._saturation_adjustment(
                dp1,
                tracers.vapor,
                tracers.liquid,
                tracers.ice,
                tracers.rain,
                tracers.snow,
                tracers.graupel,
                tracers.cloud,
                hs,
                peln,
                delp,
                delz,
                q_con,
                pt,
                pkz,
                cappa,
                zvir,
                mdt,
                fast_mp_consv,
                last_step,
                akap,
                self.kmp,
            )

        if last_step and not self._adiabatic:
            # on the last step, we need the regular temperature to send
            # to the physics, but if we're staying in dynamics we need
            # to keep it as the virtual potential temperature
            self._moist_cv_last_step_stencil(
                qvapor=tracers.vapor,
                qliquid=tracers.liquid,
                qrain=tracers.rain,
                qsnow=tracers.snow,
                qice=tracers.ice,
                qgraupel=tracers.graupel,
                pt=pt,
                pkz=pkz,
                dtmp=dtmp,
                r_vir=zvir,
            )
            self._fill_cond(
                q_con=q_con,
                qliquid=tracers.liquid,
                qrain=tracers.rain,
                qsnow=tracers.snow,
                qice=tracers.ice,
                qgraupel=tracers.graupel,
            )
        else:
            # converts virtual temperature back to virtual potential temperature
            self._adjust_divide(pkz, pt)
