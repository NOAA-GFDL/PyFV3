from pathlib import Path

import pyfv3.initialization.analytic_init as ai
from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    NullComm,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.grid import GridData, MetricTerms
from pyfv3 import DynamicalCoreConfig, DycoreState
from pyfv3.initialization import init_utils
from pyfv3.initialization.analytic_init import AnalyticCase


# JK NOTE TODO: Just sticking things in here for now, will distribute them into their right
# places in the future.


def test_enable_wam() -> None:
    # Set up dycore config with enable_wam = True 
    # Set up dycore
    # Check that gravity is variable and grav_var and grav_var_h are used somehow?
    # maybe call to compute_geopotential?

    # Set up dycore config with enable_wam = False
    # Set up dycore
    # Check that gravity is a constant... somehow
    # maybe call to compute_geopotential?

    # TODO assert that something is different between wam_enabled = False
    # JK NOTE: I have to find out what to test.....
    
    assert False # TODO


############################ dyncore_state.py
def setup_dycore_state() -> DycoreState:
    backend = "numpy"
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=13,
        npy=13,
        npz=79,
        ntiles=6,
        nwat=6,
        dt_atmos=225,
        a_imp=1.0,
        beta=0.0,
        consv_te=False,  # not implemented, needs allreduce
        d2_bg=0.0,
        d2_bg_k1=0.2,
        d2_bg_k2=0.1,
        d4_bg=0.15,
        d_con=1.0,
        d_ext=0.0,
        dddmp=0.5,
        delt_max=0.002,
        do_sat_adj=True,
        do_vort_damp=True,
        fill=True,
        hord_dp=6,
        hord_mt=6,
        hord_tm=6,
        hord_tr=8,
        hord_vt=6,
        hydrostatic=False,
        k_split=1,
        ke_bg=0.0,
        kord_mt=9,
        kord_tm=-9,
        kord_tr=9,
        kord_wz=9,
        n_split=1,
        nord=3,
        p_fac=0.05,
        rf_fast=True,
        rf_cutoff=3000.0,
        tau=10.0,
        vtdm4=0.06,
        z_tracer=True,
        do_qa=True,
    )
    mpi_comm = NullComm(
        rank=0, total_ranks=6 * config.layout[0] * config.layout[1], fill_value=0.0
    )
    partitioner = CubedSpherePartitioner(TilePartitioner(config.layout))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=config.npx - 1,
        ny_tile=config.npy - 1,
        nz=config.npz,
        n_halo=3,
        layout=config.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    #eta_file = Path("tests/data/eta79.nc")
    eta_file = Path(__file__).resolve().parents[1] / "data" / "eta79.nc"
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)
    state = ai.init_analytic_state(
        analytic_init_case=AnalyticCase.baroclinic_instability,
        grid_data=grid_data,
        quantity_factory=quantity_factory,
        adiabatic=config.adiabatic,
        hydrostatic=config.hydrostatic,
        moist_phys=config.moist_phys,
        sw_dynamics=config.sw_dynamics,
        comm=communicator,
    )
    return state


def test_dycore_state_has_wam_attributes():
    # Check that the new attributes exist.
    dycore_state = setup_dycore_state()
    assert hasattr(dycore_state, "grav_var")
    assert hasattr(dycore_state, "grav_var_h")
    # TODO: Is this really useful as a test?


############################ dyn_core.py
def test_average_gravity() -> None:
    # Check that average gravity values are reasonable
    # Use
    
    #def average_gravity(grav_var: FloatField, grav_var_h: FloatField):
    #"""
    #Args:
    #    grav_var (out): gravity field
    #    grav_var_h (in): gravity value at height
    #"""
    #with computation(FORWARD), interval(...):
    #    grav_var[0, 0, 0] = 0.5*(grav_var_h[0, 0, 0] + grav_var_h[0, 0, 1])
    assert False # TODO

def test_compute_geopotential() -> None:
    # Check that the change from constants.GRAV to grav_var_h are reasonable.
    
    #def compute_geopotential(zh: FloatField, gz: FloatField, grav_var_h: FloatField):
    #with computation(PARALLEL), interval(...):
    #    gz = zh * grav_var_h
    assert False # TODO

def test_p_grad_c_stencil() -> None:
    # Check that
    # 1. addition of average_gravity stencil and
    # 2. addition of grav_var_h parameter in self._compute_geopotential_stencil
    # still produces reasonable results
    
    assert False # TODO

############################ fv_dynamics.py

def test_init_gravity() -> None:
    # Check that init_gravity sets 3d grav_var to the constant GRAV for all vals

    assert False # TODO

def test_init_gravity_h() -> None:
    # Check that init_gravity sets 3d grav_var_h to constant GRAV for all vals
    assert False # TODO

def test_adjust_gravity() -> None:
    # Check that adjust_gravity sets grav_var and grav_var_h are set appropriately
    # with computation(FORWARD), interval(-1,None):
    #     newrad = RADIUS + (phis/GRAV)
    #     grav_var_h = GRAV*(RADIUS**2)/newrad**2
    # with computation(BACKWARD), interval(...):
    #    newrad = newrad - delz
    #    grav_var_h = GRAV*(RADIUS**2)/newrad**2
    #    grav_var = 0.5*(grav_var_h[0, 0, 1] + grav_var_h[0, 0, 0])

    assert False # TODO

# TODO JK NOTE to self --- checkout log_on_rank_0 for values that might be useful for test (possibly)
