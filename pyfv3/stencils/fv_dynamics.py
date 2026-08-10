from datetime import timedelta

import pyfv3.stencils.moist_cv as moist_cv
from ndsl import (
    NDSLRuntime,
    Quantity,
    QuantityFactory,
    StencilFactory,
    WrappedHaloUpdater,
)
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    K_DIM,
    K_INTERFACE_DIM,
    KAPPA,
    NQ,
    ZVIR,
)
from ndsl.dsl.dace.orchestration import orchestrate
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, interval
from ndsl.dsl.typing import (
    NDSL_GLOBAL_PRECISION,
    Float,
    Float64,
    FloatField,
    FloatField64,
    FloatFieldIJ64,
)
from ndsl.grid import DampingCoefficients, GridData
from ndsl.performance import Timer
from ndsl.stencils.basic_operations import copy
from ndsl.stencils.c2l_ord import CubedToLatLon
from ndsl.typing import Communicator
from pyfv3._config import DynamicalCoreConfig
from pyfv3.dycore_state import DycoreState
from pyfv3.optimization import get_optimization_config
from pyfv3.stencils import fvtp2d, tracer_2d_1l
from pyfv3.stencils.compute_total_energy import ComputeTotalEnergy
from pyfv3.stencils.del2cubed import HyperdiffusionDamping
from pyfv3.stencils.dyn_core import AcousticDynamics
from pyfv3.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from pyfv3.stencils.remapping import LagrangianToEulerian
from pyfv3.stencils.remapping_GEOS import LagrangianToEulerian_GEOS
from pyfv3.tracers import FVTracers, FVTracersAxisName
from pyfv3.version import IS_GEOS


class DryMassRoundOff(NDSLRuntime):
    def __init__(
        self,
        comm: Communicator,
        quantity_factory: QuantityFactory,
        stencil_factory: StencilFactory,
        state: DycoreState,
        hydrostatic: bool,
    ) -> None:
        super().__init__(stencil_factory)

        self._psx_2d = self.make_local(
            quantity_factory,
            [I_DIM, J_DIM],
            dtype=Float64,
            allow_mismatch_float_precision=True,
        )
        # This is a quantity because it is used _outside_ of
        # DryMassRoundOff. It should be an output
        self.dpx = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            "unknown",
            dtype=Float64,
            allow_mismatch_float_precision=True,
        )
        self._dpx0_2d = self.make_local(
            quantity_factory,
            [I_DIM, J_DIM],
            dtype=Float64,
            allow_mismatch_float_precision=True,
        )

        self._reset = stencil_factory.from_origin_domain(
            DryMassRoundOff._reset_stencil,
            origin=stencil_factory.grid_indexing.origin_compute(),
            domain=stencil_factory.grid_indexing.domain_compute(),
        )
        self._apply_psx_to_pe = stencil_factory.from_origin_domain(
            DryMassRoundOff._apply_psx_to_pe_stencil,
            origin=stencil_factory.grid_indexing.origin_compute(),
            domain=stencil_factory.grid_indexing.domain_compute(),
        )
        self._apply_dpx_to_psx = stencil_factory.from_origin_domain(
            DryMassRoundOff._apply_dpx_to_psx_stencil,
            origin=stencil_factory.grid_indexing.origin_compute(),
            domain=stencil_factory.grid_indexing.domain_compute(),
        )

        halo_spec = quantity_factory.get_quantity_halo_spec(
            dims=[I_DIM, J_DIM, K_INTERFACE_DIM],
            n_halo=stencil_factory.grid_indexing.n_halo,
            dtype=Float,
        )
        self._pe_halo_updater = WrappedHaloUpdater(
            comm.get_scalar_halo_updater([halo_spec]),
            state,
            ["pe"],
        )

        self._hydrostatic = hydrostatic

    @staticmethod
    def _reset_stencil(
        dpx: FloatField64,
        psx_2d: FloatFieldIJ64,
        pe: FloatField,
    ):
        with computation(PARALLEL), interval(...):
            dpx = 0.0
        with computation(FORWARD), interval(-1, None):
            psx_2d = pe[0, 0, 1]

    @staticmethod
    def _apply_dpx_to_psx_stencil(
        dpx: FloatField64,
        dpx0_2d: FloatFieldIJ64,
        psx_2d: FloatFieldIJ64,
    ):
        with computation(FORWARD), interval(0, 1):
            dpx0_2d = dpx

        with computation(FORWARD), interval(1, None):
            dpx0_2d += dpx

        with computation(FORWARD), interval(0, 1):
            psx_2d += psx_2d + dpx0_2d

    @staticmethod
    def _apply_psx_to_pe_stencil(
        psx_2d: FloatFieldIJ64,
        pe: FloatField,
    ):
        with computation(FORWARD), interval(-1, None):
            pe[0, 0, 1] = psx_2d

    def reset(self, pe: FloatField):
        self._reset(dpx=self.dpx, psx_2d=self._psx_2d, pe=pe)

    def apply(self, pe: FloatField):
        self._apply_dpx_to_psx(self.dpx, self._dpx0_2d, self._psx_2d)
        self._pe_halo_updater.update()
        self._apply_psx_to_pe(self._psx_2d, pe)


def _increment_stencil(
    value: FloatField,
    increment: FloatField,
):
    with computation(PARALLEL), interval(...):
        value += increment


def _copy_cast_defn(
    q_in_64: FloatField64,
    q_out: FloatField,
):
    with computation(PARALLEL), interval(...):
        q_out = q_in_64


def pt_to_potential_density_pt(
    pkz: FloatField,
    dp_initial: FloatField,
    q_con: FloatField,
    pt: FloatField,
):
    """
    Args:
        pkz (in):
        dp_initial (in):
        q_con (in):
        pt (out): temperature when input, "potential density temperature" when output
    """
    # TODO: why and how is pt being adjusted? update docstring and/or name
    # TODO: split pt into two variables for different in/out meanings
    with computation(PARALLEL), interval(...):
        pt = pt * (1.0 + dp_initial) * (1.0 - q_con) / pkz


def omega_from_w(
    delp: FloatField,
    delz: FloatField,
    w: FloatField,
    omega: FloatField,
):
    """
    Args:
        delp (in): vertical layer thickness in Pa
        delz (in): vertical layer thickness in m
        w (in): vertical wind in m/s
        omega (out): vertical wind in Pa/s
    """
    with computation(PARALLEL), interval(...):
        omega = delp / delz * w


def _reset_to_zero(field: FloatField):
    with computation(PARALLEL), interval(...):
        field = 0


class DynamicalCore(NDSLRuntime):
    """
    Corresponds to fv_dynamics in original Fortran sources.
    """

    def __init__(
        self,
        comm: Communicator,
        grid_data: GridData,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        damping_coefficients: DampingCoefficients,
        config: DynamicalCoreConfig,
        phis: Quantity,
        state: DycoreState,
        timestep: timedelta,
    ):
        """
        Args:
            comm: object for cubed sphere or tile inter-process communication
            grid_data: metric terms defining the model grid
            stencil_factory: creates stencils
            damping_coefficients: damping configuration/constants
            config: configuration of dynamical core, for example as would be set by
                the namelist in the Fortran model
            phis: surface geopotential height
            state: model state
            exclude_tracer: List of named tracer to be excluded from the Advection,
                and Remapping schemes
            timestep: model timestep
        """

        oconfig = get_optimization_config(stencil_factory.backend)
        super().__init__(stencil_factory, oconfig)

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="step_dynamics",
            dace_compiletime_args=["state", "timer"],
            optimization_config=oconfig,
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="compute_preamble",
            dace_compiletime_args=["state"],
            optimization_config=oconfig,
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_compute",
            dace_compiletime_args=["state", "timer"],
            optimization_config=oconfig,
        )

        if timestep == timedelta(seconds=0):
            raise RuntimeError(
                "Bad dynamical core configuration: the atmospheric timestep is 0 seconds!"
            )
        # nested and stretched_grid are options in the Fortran code which we
        # have not implemented, so they are hard-coded here.
        nested = False
        stretched_grid = False
        grid_indexing = stencil_factory.grid_indexing
        if not config.moist_phys:
            raise NotImplementedError(
                "Dynamical core (fv_dynamics): fvsetup is only implemented for moist_phys=true."
            )
        if config.nwat not in [0, 6]:
            raise NotImplementedError(
                "Dynamical core (fv_dynamics):"
                f" nwat=={config.nwat} is not implemented."
                " Only nwat=0 or 6 has been implemented."
            )

        if config.nwat == 6:
            # Implemented dynamics options require those tracers to be present at minima
            # this is a more granular list than carried by the `nwat` single integer
            # but cover the same topic
            required_tracers = [
                "vapor",
                "liquid",
                "rain",
                "snow",
                "ice",
                "graupel",
                "cloud",
            ]
            if not all(n in FVTracers.mapping.keys() for n in required_tracers):
                raise NotImplementedError(
                    "Dynamical core (fv_dynamics):"
                    " missing required tracers. Dynamics requires:\n"
                    f" {required_tracers}\n"
                    "but only the following where given:\n"
                    f" {FVTracers.mapping.keys()}"
                )

        self._comm = comm
        self.comm_rank = comm.rank
        self.grid_data = grid_data
        self.grid_indexing = grid_indexing
        self._da_min = damping_coefficients.da_min
        self.config = config

        self.dry_mass_control = DryMassRoundOff(
            comm=comm,
            quantity_factory=quantity_factory,
            stencil_factory=stencil_factory,
            state=state,
            hydrostatic=self.config.hydrostatic,
        )

        tracer_transport = fvtp2d.FiniteVolumeTransport(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            hord=config.hord_tr,
        )

        if FVTracersAxisName not in quantity_factory.sizer.data_dimensions:
            raise RuntimeError(
                "FV Dynamics requires FVTracers to be registered - see `pyfv3.tracers`"
            )

        # Locals
        self._wsd = self.make_local(quantity_factory, [I_DIM, J_DIM])
        self._dp_initial = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._cvm = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

        # TODO: this is a true Local, but defining at such breaks `pt` in orchestration
        self._te0_2d = quantity_factory.zeros([I_DIM, J_DIM], "")

        # Build advection stencils
        self.tracer_advection = tracer_2d_1l.TracerAdvection(
            stencil_factory,
            quantity_factory,
            tracer_transport,
            self.grid_data,
            comm,
            state.tracers,
        )
        self._ak = grid_data.ak
        self._bk = grid_data.bk
        self._phis = phis
        self._ptop = self.grid_data.ptop
        self._pfull = grid_data.p
        self._fv_setup_stencil = stencil_factory.from_origin_domain(
            moist_cv.fv_setup,
            externals={
                "nwat": self.config.nwat,
                "moist_phys": self.config.moist_phys,
                "i_vapor": FVTracers.index("vapor"),
                "i_liquid": FVTracers.index("liquid") if self.config.nwat == 6 else -1,
                "i_rain": FVTracers.index("rain") if self.config.nwat == 6 else -1,
                "i_ice": FVTracers.index("ice") if self.config.nwat == 6 else -1,
                "i_snow": FVTracers.index("snow") if self.config.nwat == 6 else -1,
                "i_graupel": (
                    FVTracers.index("graupel") if self.config.nwat == 6 else -1
                ),
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._pt_to_potential_density_pt = stencil_factory.from_origin_domain(
            pt_to_potential_density_pt,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._omega_from_w = stencil_factory.from_origin_domain(
            omega_from_w,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._copy_stencil = stencil_factory.from_origin_domain(
            copy,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(),
        )
        self._copy_domain = stencil_factory.from_origin_domain(
            copy,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self.acoustic_dynamics = AcousticDynamics(
            comm=comm,
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            grid_type=config.grid_type,
            nested=nested,
            stretched_grid=stretched_grid,
            config=self.config.acoustic_dynamics,
            phis=self._phis,
            state=state,
        )
        self._hyperdiffusion = HyperdiffusionDamping(
            stencil_factory,
            quantity_factory,
            damping_coefficients,
            grid_data.rarea,
            self.config.nf_omega,
        )
        self._cubed_to_latlon = CubedToLatLon(
            state,
            stencil_factory,
            quantity_factory,
            grid_data,
            self.config.grid_type,
            config.c2l_ord,
            comm,
        )
        self._cappa = self.acoustic_dynamics.cappa

        if not (not self.config.inline_q and NQ != 0):
            raise NotImplementedError(
                "Dynamical core (fv_dynamics):tracer_2d not implemented. z_tracer available"
            )
        self._adjust_tracer_mixing_ratio = AdjustNegativeTracerMixingRatio(
            stencil_factory,
            quantity_factory=quantity_factory,
            check_negative=self.config.check_negative,
            hydrostatic=self.config.hydrostatic,
        )

        self._compute_total_energy = ComputeTotalEnergy(
            config=config,
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
        )

        if IS_GEOS:
            self._lagrangian_to_eulerian_GEOS = LagrangianToEulerian_GEOS(
                stencil_factory=stencil_factory,
                quantity_factory=quantity_factory,
                config=config.remapping,
                comm=comm,
                grid_data=grid_data,
                pfull=self._pfull,
                adiabatic=config.adiabatic,
                nwat=self.config.nwat,
            )

        else:
            self._lagrangian_to_eulerian_obj = LagrangianToEulerian(
                stencil_factory=stencil_factory,
                quantity_factory=quantity_factory,
                config=config.remapping,
                area_64=grid_data.area_64,
                pfull=self._pfull,
                nwat=self.config.nwat,
            )

        full_xyz_spec = quantity_factory.get_quantity_halo_spec(
            dims=[I_DIM, J_DIM, K_DIM],
            n_halo=grid_indexing.n_halo,
            dtype=Float,
        )
        self._omega_halo_updater = WrappedHaloUpdater(
            comm.get_scalar_halo_updater([full_xyz_spec]), state, ["omga"], comm=comm
        )
        self._n_split = config.n_split
        self._k_split = config.k_split
        self._conserve_total_energy = config.consv_te
        self._timestep = timestep.total_seconds()

        # At 32-bit precision we still need
        self._f32_correction = NDSL_GLOBAL_PRECISION == 32
        if self._f32_correction:
            self._mfx_f64 = quantity_factory.zeros(
                dims=[I_INTERFACE_DIM, J_DIM, K_DIM],
                units="unknown",
                dtype=Float64,
                allow_mismatch_float_precision=True,
            )
            self._mfy_f64 = quantity_factory.zeros(
                dims=[I_DIM, J_INTERFACE_DIM, K_DIM],
                units="unknown",
                dtype=Float64,
                allow_mismatch_float_precision=True,
            )
            self._cx_f64 = quantity_factory.zeros(
                dims=[I_INTERFACE_DIM, J_DIM, K_DIM],
                units="unknown",
                dtype=Float64,
                allow_mismatch_float_precision=True,
            )
            self._cy_f64 = quantity_factory.zeros(
                dims=[I_DIM, J_INTERFACE_DIM, K_DIM],
                units="unknown",
                dtype=Float64,
                allow_mismatch_float_precision=True,
            )
        self._mfx_local = quantity_factory.zeros(
            dims=[I_INTERFACE_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._mfy_local = quantity_factory.zeros(
            dims=[I_DIM, J_INTERFACE_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._cx_local = quantity_factory.zeros(
            dims=[I_INTERFACE_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._cy_local = quantity_factory.zeros(
            dims=[I_DIM, J_INTERFACE_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._reset_I_interface = stencil_factory.from_origin_domain(
            func=_reset_to_zero,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(1, 0, 0)),
        )
        self._reset_J_interface = stencil_factory.from_origin_domain(
            func=_reset_to_zero,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(0, 1, 0)),
        )
        self._increment = stencil_factory.from_origin_domain(
            func=_increment_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(1, 1, 0)),
        )
        self._copy_cast = stencil_factory.from_origin_domain(
            func=_copy_cast_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(add=(1, 1, 0)),
        )

    def step_dynamics(self, state: DycoreState, timer: Timer) -> None:
        """
        Step the model state forward by one timestep.

        Args:
            state: model prognostic state and inputs
            timer: keep time of model sections
        """
        self._compute(state, timer)

    def compute_preamble(self, state: DycoreState) -> None:
        if self.config.hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")

        # Reset fluxes
        self._reset_I_interface(state.mfxd)
        self._reset_I_interface(state.cxd)
        self._reset_J_interface(state.mfyd)
        self._reset_J_interface(state.cyd)

        self._fv_setup_stencil(
            state.tracers,
            state.q_con,
            self._cvm,
            state.pkz,
            state.pt,
            self._cappa,
            state.delp,
            state.delz,
            self._dp_initial,
        )

        # Compute total energy
        if self.config.consv_te > 0.0:
            self._compute_total_energy(
                hs=state.phis,
                delp=state.delp,
                delz=state.delz,
                qc=self._dp_initial,
                pt=state.pt,
                u=state.u,
                v=state.v,
                w=state.w,
                tracers=state.tracers,
                te_2d=self._te0_2d,
            )

        # Rayleigh fast
        if (
            not self.config.hydrostatic
            and not self.config.acoustic_dynamics.rf_fast
            and self.config.acoustic_dynamics.tau > 0
        ):
            raise NotImplementedError(
                "Dynamical Core (fv_dynamics): Rayleigh Friction is not implemented."
            )

        # Adjust pt
        if self.config.adiabatic:
            raise NotImplementedError(
                "Dynamical Core (fv_dynamics): Adiabatic pt adjust is not implemented."
            )
        else:
            if self.config.hydrostatic:
                raise NotImplementedError(
                    "Dynamical Core (fv_dynamics): Hydrostatic pt adjust is not implemented."
                )
            else:
                self._pt_to_potential_density_pt(
                    state.pkz,
                    self._dp_initial,
                    state.q_con,
                    state.pt,
                )

        self.dry_mass_control.reset(pe=state.pe)

    def __call__(self, *args, **kwargs) -> None:
        self.step_dynamics(*args, **kwargs)

    def _compute(self, state: DycoreState, timer: Timer) -> None:
        self.compute_preamble(state)

        for k_split in range(self._k_split):
            n_map = k_split + 1
            last_step = k_split == self._k_split - 1
            # TODO: why are we copying delp to dp1? what is dp1?
            self._copy_stencil(
                state.delp,
                self._dp_initial,
            )

            with timer.clock("DynCore"):
                self.acoustic_dynamics(
                    state=state,
                    mfxd=self._mfx_f64 if self._f32_correction else self._mfx_local,
                    mfyd=self._mfy_f64 if self._f32_correction else self._mfy_local,
                    cxd=self._cx_f64 if self._f32_correction else self._cx_local,
                    cyd=self._cy_f64 if self._f32_correction else self._cy_local,
                    dpx=self.dry_mass_control.dpx,
                    wsd=self._wsd,
                    timestep=self._timestep / self._k_split,
                    n_map=n_map,
                )
                if self._f32_correction:
                    self._copy_cast(self._mfx_f64, self._mfx_local)
                    self._copy_cast(self._mfy_f64, self._mfy_local)
                    self._copy_cast(self._cx_f64, self._cx_local)
                    self._copy_cast(self._cy_f64, self._cy_local)
                if last_step and self.config.hydrostatic:
                    self.dry_mass_control.apply(state.pe)
            if self.config.z_tracer:
                with timer.clock("TracerAdvection"):
                    self.tracer_advection(
                        state.tracers,
                        self._dp_initial,
                        x_mass_flux=self._mfx_local,
                        y_mass_flux=self._mfy_local,
                        x_courant=self._cx_local,
                        y_courant=self._cy_local,
                    )
            else:
                raise NotImplementedError("z_tracer=False is not implemented")

            # 1 is shallow water model, don't need vertical remapping
            # 2 and 3 are also simple baroclinic models that don't need
            # vertical remapping. > 4 implies this is a full physics model
            if self.grid_indexing.domain[2] > 4:
                # nq is actually given by ncnst - pnats,
                # where those are given in atmosphere.F90 by:
                # ncnst = Atm(mytile)%ncnst
                # pnats = Atm(mytile)%flagstruct%pnats
                # here we hard-coded it because 8 is the only supported value,
                # refactor this later!

                # do_omega = self.namelist.hydrostatic and last_step
                # TODO: Determine a better way to do this, polymorphic fields perhaps?
                # issue is that set_val in map_single expects a 3D field for the
                # "surface" array
                with timer.clock("Remapping"):
                    if IS_GEOS:
                        self._lagrangian_to_eulerian_GEOS(
                            tracers=state.tracers,
                            pt=state.pt,
                            delp=state.delp,
                            delz=state.delz,
                            peln=state.peln,
                            u=state.u,
                            v=state.v,
                            w=state.w,
                            mfx=self._mfx_local,
                            mfy=self._mfy_local,
                            cx=self._cx_local,
                            cy=self._cy_local,
                            cappa=self._cappa,
                            q_con=state.q_con,
                            pkz=state.pkz,
                            pk=state.pk,
                            pe=state.pe,
                            hs=state.phis,
                            te0_2d=self._te0_2d,
                            ps=state.ps,
                            wsd=self._wsd,
                            ak=self._ak,
                            bk=self._bk,
                            dp1=self._dp_initial,
                            ptop=self._ptop,
                            akap=KAPPA,
                            zvir=ZVIR,
                            last_step=last_step,
                            consv_te=self._conserve_total_energy,
                            mdt=self._timestep / self._k_split,
                        )
                    else:
                        # TODO: When NQ=9, we shouldn't need to pass qcld explicitly
                        #       since it's in self.tracers. It should not be an issue
                        #       since we don't have self.tracers & qcld computation
                        #       at the same time
                        #       When NQ=8, we do need qcld passed explicitely
                        self._lagrangian_to_eulerian_obj(
                            state.tracers,
                            state.pt,
                            state.delp,
                            state.delz,
                            state.peln,
                            state.u,
                            state.v,
                            state.w,
                            self._cappa,
                            state.q_con,
                            state.pkz,
                            state.pk,
                            state.pe,
                            state.phis,
                            state.ps,
                            self._wsd,
                            self._ak,
                            self._bk,
                            self._dp_initial,
                            self._ptop,
                            KAPPA,
                            ZVIR,
                            last_step,
                            self._conserve_total_energy,
                            self._timestep / self._k_split,
                        )
                # TODO: can we pull this block out of the loop intead of
                # using an if-statement?

                # Update state fluxes and courant number
                self._increment(state.mfxd, self._mfx_local)
                self._increment(state.mfyd, self._mfy_local)
                self._increment(state.cxd, self._cx_local)
                self._increment(state.cyd, self._cy_local)

                if last_step:
                    if not self.config.hydrostatic:
                        # TODO: GFDL should implement the "vulcan omega" update,
                        # use hydrostatic omega instead of this conversion
                        self._omega_from_w(
                            state.delp,
                            state.delz,
                            state.w,
                            state.omga,
                        )
                    if self.config.nf_omega > 0:
                        self._omega_halo_updater.update()
                        self._hyperdiffusion(state.omga, Float(0.18) * self._da_min)

        if self.config.nwat >= 6:
            self._adjust_tracer_mixing_ratio(
                state.tracers,
                state.pt,
                state.delp,
            )

        # convert d-grid x-wind and y-wind to
        # cell-centered zonal and meridional winds
        # TODO: make separate variables for the internal-temporary
        # usage of ua and va, and rename state.ua and state.va
        # to reflect that they are cell center
        # zonal and meridional wind
        self._cubed_to_latlon(
            state.u,
            state.v,
            state.ua,
            state.va,
        )
