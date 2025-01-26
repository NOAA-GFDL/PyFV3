from datetime import timedelta
from typing import List, Mapping, Optional

from dace.frontend.python.interface import nounroll as dace_no_unroll
from gt4py.cartesian.gtscript import PARALLEL, FORWARD, computation, interval

import pyFV3.stencils.moist_cv as moist_cv
from ndsl import Quantity, QuantityFactory, StencilFactory, WrappedHaloUpdater, is
from ndsl.checkpointer import NullCheckpointer
from ndsl.comm.mpi import MPI
from ndsl.constants import (
    KAPPA,
    NQ,
    X_DIM,
    Y_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
    ZVIR,
    Y_INTERFACE_DIM,
    X_INTERFACE_DIM,
)
from ndsl.dsl.dace.orchestration import dace_inhibitor, orchestrate
from ndsl.dsl.typing import (
    Float,
    FloatField,
    FloatField64,
    FloatFieldIJ64,
    global_set_floating_point_precision,
    NDSL_32BIT_FLOAT_TYPE,
    NDSL_64BIT_FLOAT_TYPE,
)
from ndsl.grid import DampingCoefficients, GridData
from ndsl.logging import ndsl_log
from ndsl.performance import NullTimer, Timer
from ndsl.stencils.basic_operations import copy_defn, set_value_defn
from ndsl.stencils.c2l_ord import CubedToLatLon
from ndsl.typing import Checkpointer, Communicator
from pyFV3._config import DynamicalCoreConfig
from pyFV3.dycore_state import DycoreState
from pyFV3.stencils import fvtp2d, tracer_2d_1l
from pyFV3.stencils.del2cubed import HyperdiffusionDamping
from pyFV3.stencils.dyn_core import AcousticDynamics
from pyFV3.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from pyFV3.stencils.remapping import LagrangianToEulerian
from pyFV3.stencils.remapping_GEOS import LagrangianToEulerian_GEOS
from pyFV3.stencils.compute_total_energy import ComputeTotalEnergy
from pyFV3.version import IS_GEOS


class DryMassRoundOff:
    def __init__(
        self,
        comm: Communicator,
        quantity_factory: QuantityFactory,
        stencil_factory: StencilFactory,
        state: DycoreState,
        hydrostatic: bool,
    ) -> None:
        self.psx_2d = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="unknown",
            dtype=NDSL_64BIT_FLOAT_TYPE,
            allow_mismatch_float_precision=True,
        )
        self.dpx = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=NDSL_64BIT_FLOAT_TYPE,
            allow_mismatch_float_precision=True,
        )
        self.dpx0_2d = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="unknown",
            dtype=NDSL_64BIT_FLOAT_TYPE,
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
            dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
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
        dpx: FloatField64,  # type:ignore
        psx_2d: FloatFieldIJ64,  # type:ignore
        pe: FloatField,  # type:ignore
    ):
        with computation(PARALLEL), interval(...):
            dpx = 0.0
        with computation(FORWARD), interval(-1, None):
            psx_2d = pe[0, 0, 1]

    @staticmethod
    def _apply_dpx_to_psx_stencil(
        dpx: FloatField64,  # type:ignore
        dpx0_2d: FloatFieldIJ64,  # type:ignore
        psx_2d: FloatFieldIJ64,  # type:ignore
    ):
        with computation(FORWARD), interval(0, 1):
            dpx0_2d = dpx

        with computation(FORWARD), interval(1, None):
            dpx0_2d += dpx

        with computation(FORWARD), interval(0, 1):
            psx_2d += psx_2d + dpx0_2d

    @staticmethod
    def _apply_psx_to_pe_stencil(
        psx_2d: FloatFieldIJ64,  # type:ignore
        pe: FloatField,  # type:ignore
    ):
        with computation(FORWARD), interval(-1, None):
            pe[0, 0, 1] = psx_2d

    def reset(self, pe: FloatField):  # type:ignore
        self._reset(dpx=self.dpx, psx_2d=self.psx_2d, pe=pe)

    def apply(self, pe: FloatField):  # type:ignore
        self._apply_dpx_to_psx(self.dpx, self.dpx0_2d, self.psx_2d)
        self._pe_halo_updater.update()
        self._apply_psx_to_pe(self.psx_2d, pe)


def _increment_stencil(
    value: FloatField,  # type:ignore
    increment: FloatField,  # type:ignore
):
    with computation(PARALLEL), interval(...):
        value += increment


def _copy_cast_defn(
    q_in_64: FloatField64,  # type:ignore
    q_out: FloatField,  # type:ignore
):
    with computation(PARALLEL), interval(...):
        q_out = q_in_64


def pt_to_potential_density_pt(
    pkz: FloatField,  # type: ignore
    dp_initial: FloatField,  # type: ignore
    q_con: FloatField,  # type: ignore
    pt: FloatField,  # type: ignore
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
    delp: FloatField,  # type: ignore
    delz: FloatField,  # type: ignore
    w: FloatField,  # type: ignore
    omega: FloatField,  # type: ignore
):
    """
    Args:
        delp (in): vertical layer thickness in Pa
        delz (in): vertical layer thickness in m
        w (in): vertical wind in m/s
        omga (out): vertical wind in Pa/s
    """
    with computation(PARALLEL), interval(...):
        omega = delp / delz * w


def fvdyn_temporaries(
    quantity_factory: QuantityFactory,
) -> Mapping[str, Quantity]:
    tmps = {}
    for name in ["te0_2d", "wsd"]:
        quantity = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        tmps[name] = quantity
    for name in ["dp1", "cvm"]:
        quantity = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        tmps[name] = quantity
    return tmps


@dace_inhibitor
def log_on_rank_0(msg: str):
    """Print when rank is 0 - outside of DaCe critical path"""
    if not MPI or MPI.COMM_WORLD.Get_rank() == 0:
        ndsl_log.info(msg)


class DynamicalCore:
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
        exclude_tracers: List[str],
        timestep: timedelta,
        checkpointer: Optional[Checkpointer] = None,
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
            checkpointer: if given, used to perform operations on model data
                at specific points in model execution, such as testing against
                reference data
        """
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="step_dynamics",
            dace_compiletime_args=["state", "timer"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="compute_preamble",
            dace_compiletime_args=["state", "is_root_rank"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_compute",
            dace_compiletime_args=["state", "timer"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_fvdynamics",
            dace_compiletime_args=["state", "tag"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_remapping_in",
            dace_compiletime_args=[
                "state",
            ],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_remapping_out",
            dace_compiletime_args=["state"],
        )

        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_tracer_advection_in",
            dace_compiletime_args=["state"],
        )
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="_checkpoint_tracer_advection_out",
            dace_compiletime_args=["state"],
        )
        if timestep == timedelta(seconds=0):
            raise RuntimeError(
                "Bad dynamical core configuration:"
                " the atmospheric timestep is 0 seconds!"
            )
        # nested and stretched_grid are options in the Fortran code which we
        # have not implemented, so they are hard-coded here.
        self.call_checkpointer = checkpointer is not None
        if checkpointer is None:
            self.checkpointer: Checkpointer = NullCheckpointer()
        else:
            self.checkpointer = checkpointer
        nested = False
        stretched_grid = False
        grid_indexing = stencil_factory.grid_indexing
        if not config.moist_phys:
            raise NotImplementedError(
                "Dynamical core (fv_dynamics):"
                " fvsetup is only implemented for moist_phys=true."
            )
        if config.nwat != 6:
            raise NotImplementedError(
                "Dynamical core (fv_dynamics):"
                f" nwat=={config.nwat} is not implemented."
                " Only nwat=6 has been implemented."
            )

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
        if not all(n in state.tracers.names() for n in required_tracers):
            raise NotImplementedError(
                "Dynamical core (fv_dynamics):"
                " missing required tracers. Dynamics requires:\n"
                f" {required_tracers}\n"
                "but only the following where given:\n"
                f" {state.tracers.names()}"
            )

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

        temporaries = fvdyn_temporaries(quantity_factory)
        self._te0_2d = temporaries["te0_2d"]
        self._wsd = temporaries["wsd"]
        self._dp_initial = temporaries["dp1"]
        self._cvm = temporaries["cvm"]

        # Build advection stencils
        self.tracer_advection = tracer_2d_1l.TracerAdvection(
            stencil_factory,
            quantity_factory,
            tracer_transport,
            self.grid_data,
            comm,
            state.tracers,
            exclude_tracers=exclude_tracers,
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
            copy_defn,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(),
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
            wsd=self._wsd,
            state=state,
            checkpointer=checkpointer,
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
                "Dynamical core (fv_dynamics):"
                "tracer_2d not implemented. z_tracer available"
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
                nq=NQ,
                pfull=self._pfull,
                tracers=state.tracers,
                adiabatic=config.adiabatic,
            )

        else:
            self._lagrangian_to_eulerian_obj = LagrangianToEulerian(
                stencil_factory=stencil_factory,
                quantity_factory=quantity_factory,
                config=config.remapping,
                area_64=grid_data.area_64,
                pfull=self._pfull,
                tracers=state.tracers,
                exclude_tracers=exclude_tracers,
                checkpointer=checkpointer,
            )

        full_xyz_spec = quantity_factory.get_quantity_halo_spec(
            dims=[X_DIM, Y_DIM, Z_DIM],
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
        self._f32_correction = (
            global_set_floating_point_precision() == NDSL_32BIT_FLOAT_TYPE
        )
        if self._f32_correction:
            self._mfx_f64 = quantity_factory.zeros(
                dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=NDSL_64BIT_FLOAT_TYPE,
                allow_mismatch_float_precision=True,
            )
            self._mfy_f64 = quantity_factory.zeros(
                dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
                units="unknown",
                dtype=NDSL_64BIT_FLOAT_TYPE,
                allow_mismatch_float_precision=True,
            )
            self._cx_f64 = quantity_factory.zeros(
                dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=NDSL_64BIT_FLOAT_TYPE,
                allow_mismatch_float_precision=True,
            )
            self._cy_f64 = quantity_factory.zeros(
                dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
                units="unknown",
                dtype=NDSL_64BIT_FLOAT_TYPE,
                allow_mismatch_float_precision=True,
            )
        self._mfx_local = quantity_factory.zeros(
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._mfy_local = quantity_factory.zeros(
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._cx_local = quantity_factory.zeros(
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._cy_local = quantity_factory.zeros(
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )
        self._set_value = stencil_factory.from_origin_domain(
            func=set_value_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._increment = stencil_factory.from_origin_domain(
            func=_increment_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._copy_cast = stencil_factory.from_origin_domain(
            func=_copy_cast_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    # See divergence_damping.py, _get_da_min for explanation of this function
    @dace_inhibitor
    def _get_da_min(self) -> Float:  # type: ignore
        return Float(self._da_min)

    def _checkpoint_fvdynamics(self, state: DycoreState, tag: str):
        if self.call_checkpointer:
            self.checkpointer(
                f"FVDynamics-{tag}",
                u=state.u,
                v=state.v,
                w=state.w,
                delz=state.delz,
                # ua is not checked as its halo values differ from Fortran,
                # this can be re-enabled if no longer comparing to Fortran, if the
                # Fortran is updated to match the Python, or if the checkpointer
                # can check only the compute domain values
                # ua=state.ua,
                va=state.va,
                uc=state.uc,
                vc=state.vc,
                qvapor=state.tracers["vapor"],
            )

    def _checkpoint_remapping_in(
        self,
        state: DycoreState,
    ):
        if self.call_checkpointer:
            self.checkpointer(
                "Remapping-In",
                pt=state.pt,
                delp=state.delp,
                delz=state.delz,
                peln=state.peln.transpose(
                    [X_DIM, Z_INTERFACE_DIM, Y_DIM]
                ),  # [x, z, y] fortran data
                u=state.u,
                v=state.v,
                w=state.w,
                ua=state.ua,
                va=state.va,
                cappa=self._cappa,
                pk=state.pk,
                pe=state.pe.transpose(
                    [X_DIM, Z_INTERFACE_DIM, Y_DIM]
                ),  # [x, z, y] fortran data
                phis=state.phis,
                te_2d=self._te0_2d,
                ps=state.ps,
                wsd=self._wsd,
                omga=state.omga,
                dp1=self._dp_initial,
            )

    def _checkpoint_remapping_out(
        self,
        state: DycoreState,
    ):
        if self.call_checkpointer:
            self.checkpointer(
                "Remapping-Out",
                pt=state.pt,
                delp=state.delp,
                delz=state.delz,
                peln=state.peln.transpose(
                    [X_DIM, Z_INTERFACE_DIM, Y_DIM]
                ),  # [x, z, y] fortran data
                u=state.u,
                v=state.v,
                w=state.w,
                cappa=self._cappa,
                pkz=state.pkz,
                pk=state.pk,
                pe=state.pe.transpose(
                    [X_DIM, Z_INTERFACE_DIM, Y_DIM]
                ),  # [x, z, y] fortran data
                dp1=self._dp_initial,
            )

    def _checkpoint_tracer_advection_in(
        self,
        state: DycoreState,
    ):
        if self.call_checkpointer:
            self.checkpointer(
                "Tracer2D1L-In",
                dp1=self._dp_initial,
                mfxd=state.mfxd,
                mfyd=state.mfyd,
                cxd=state.cxd,
                cyd=state.cyd,
            )

    def _checkpoint_tracer_advection_out(
        self,
        state: DycoreState,
    ):
        if self.call_checkpointer:
            self.checkpointer(
                "Tracer2D1L-Out",
                dp1=self._dp_initial,
                mfxd=state.mfxd,
                mfyd=state.mfyd,
                cxd=state.cxd,
                cyd=state.cyd,
            )

    def step_dynamics(
        self,
        state: DycoreState,
        timer: Timer = NullTimer(),
    ):
        """
        Step the model state forward by one timestep.

        Args:
            timer: keep time of model sections
            state: model prognostic state and inputs
        """
        self._checkpoint_fvdynamics(state=state, tag="In")
        self._compute(state, timer)
        self._checkpoint_fvdynamics(state=state, tag="Out")

    def compute_preamble(self, state: DycoreState, is_root_rank: bool):
        if self.config.hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")
        if __debug__:
            log_on_rank_0("FV Setup")

        # Reset fluxes
        self._set_value(state.mfxd, Float(0.0))
        self._set_value(state.mfyd, Float(0.0))
        self._set_value(state.cxd, Float(0.0))
        self._set_value(state.cyd, Float(0.0))

        self._fv_setup_stencil(
            state.tracers["vapor"],
            state.tracers["liquid"],
            state.tracers["rain"],
            state.tracers["snow"],
            state.tracers["ice"],
            state.tracers["graupel"],
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
            not self.config.acoustic_dynamics.rf_fast
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
                    "Dynamical Core (fv_dynamics): Hydrostatic pt adjust"
                    " is not implemented."
                )
            else:
                self._pt_to_potential_density_pt(
                    state.pkz,
                    self._dp_initial,
                    state.q_con,
                    state.pt,
                )

        self.dry_mass_control.reset(pe=state.pe)

    def __call__(self, *args, **kwargs):
        return self.step_dynamics(*args, **kwargs)

    def _compute(self, state: DycoreState, timer: Timer):
        last_step = False
        self.compute_preamble(
            state,
            is_root_rank=self.comm_rank == 0,
        )

        for k_split in dace_no_unroll(range(self._k_split)):
            n_map = k_split + 1
            last_step = k_split == self._k_split - 1
            # TODO: why are we copying delp to dp1? what is dp1?
            self._copy_stencil(
                state.delp,
                self._dp_initial,
            )
            if __debug__:
                log_on_rank_0("DynCore")
            with timer.clock("DynCore"):
                self.acoustic_dynamics(
                    state=state,
                    mfxd=self._mfx_f64 if self._f32_correction else self._mfx_local,
                    mfyd=self._mfy_f64 if self._f32_correction else self._mfy_local,
                    cxd=self._cx_f64 if self._f32_correction else self._cx_local,
                    cyd=self._cy_f64 if self._f32_correction else self._cy_local,
                    dpx=self.dry_mass_control.dpx,
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
                if __debug__:
                    log_on_rank_0("TracerAdvection")
                with timer.clock("TracerAdvection"):
                    self._checkpoint_tracer_advection_in(state)
                    self.tracer_advection(
                        state.tracers,
                        self._dp_initial,
                        x_mass_flux=self._mfx_local,
                        y_mass_flux=self._mfy_local,
                        x_courant=self._cx_local,
                        y_courant=self._cy_local,
                    )
                    self._checkpoint_tracer_advection_out(state)
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
                if __debug__:
                    log_on_rank_0("Remapping")
                with timer.clock("Remapping"):
                    self._checkpoint_remapping_in(state)

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
                    self._checkpoint_remapping_out(state)
                # TODO: can we pull this block out of the loop intead of
                # using an if-statement?

                # Update state fluxes and courant number
                self._increment(state.mfxd, self._mfx_local)
                self._increment(state.mfyd, self._mfy_local)
                self._increment(state.cxd, self._cx_local)
                self._increment(state.cyd, self._cy_local)

                if last_step:
                    da_min = self._get_da_min()
                    if not self.config.hydrostatic:
                        if __debug__:
                            log_on_rank_0("Omega")
                        # TODO: GFDL should implement the "vulcan omega" update,
                        # use hydrostatic omega instead of this conversion
                        self._omega_from_w(
                            state.delp,
                            state.delz,
                            state.w,
                            state.omga,
                        )
                    if self.config.nf_omega > 0:
                        if __debug__:
                            log_on_rank_0("Del2Cubed")
                        self._omega_halo_updater.update()
                        self._hyperdiffusion(state.omga, Float(0.18) * da_min)

        if __debug__:
            log_on_rank_0("Neg Adj 3")
        self._adjust_tracer_mixing_ratio(
            state.tracers["vapor"],
            state.tracers["liquid"],
            state.tracers["rain"],
            state.tracers["snow"],
            state.tracers["ice"],
            state.tracers["graupel"],
            state.tracers["cloud"],
            state.pt,
            state.delp,
        )

        if __debug__:
            log_on_rank_0("CubedToLatLon")
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
