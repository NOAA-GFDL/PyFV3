import copy
from datetime import timedelta
from pathlib import Path
from typing import Tuple

# use numpy for now until I figure out how to use FloatField
import numpy as np  # JK TODO: Should I be using xumpy?

import pyfv3.initialization.analytic_init as ai
from ndsl import (
    CompilationConfig,
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    DaceConfig,
    GridIndexing,
    NullComm,
    Quantity,
    QuantityFactory,
    StencilConfig,
    StencilFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from ndsl.constants import GRAV, RADIUS, RDGAS, X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.gt4py import stencil
from ndsl.dsl.typing import Float
from ndsl.grid import DampingCoefficients, GridData, MetricTerms
from ndsl.stencils.basic_operations import set_value
from pyfv3 import DycoreState, DynamicalCore, DynamicalCoreConfig
from pyfv3.initialization.analytic_init import AnalyticCase
from pyfv3.stencils.dyn_core import AcousticDynamics
from pyfv3.stencils.wam import adjust_gravity, average_gravity_stencil_defn, neg_rdgas_div_gravity
from pyfv3.stencils.dyn_core import compute_geopotential


# JK NOTE TODO: Just sticking things in here for now,
# will distribute them into their right places in the future.


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

    assert False  # TODO


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
    # eta_file = Path("tests/data/eta79.nc")
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
    assert hasattr(dycore_state, "rdg_var")
    # JK TODO: Is this really useful as a test?


############################ dyn_core.py
def test_neg_rdgas_div_gravity() -> None:
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend = "numpy"

    grav_var = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="grav_var units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    rdg = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="rdg units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    init_gravity_stencil = stencil(backend=example_backend, definition=set_value)
    init_gravity_stencil(grav_var, GRAV)
    expected_grav_var_np = copy.deepcopy(grav_var.field[:])

    neg_rdgas_div_gravity_stencil = stencil(
        backend=example_backend, definition=neg_rdgas_div_gravity
    )
    neg_rdgas_div_gravity_stencil(rdg, grav_var)

    # grav_var_h should be unchanged by the stencil
    assert np.array_equal(grav_var.field[:], expected_grav_var_np)

    expected_rdg_np = -(RDGAS / expected_grav_var_np[:])
    assert np.array_equal(rdg.field[:], expected_rdg_np)


def test_average_gravity() -> None:
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend = "numpy"

    grav_var = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="grav_var units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    grav_var_h_np = np.random.random((nx, ny, nz + 1))
    expected_grav_var_h_np = copy.deepcopy(grav_var_h_np)
    grav_var_h = Quantity(
        data=grav_var_h_np,
        dims=example_dims,
        units="grav_var_h units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    average_gravity_numpy = stencil(
        backend=example_backend, definition=average_gravity_stencil_defn
    )
    average_gravity_numpy(grav_var, grav_var_h)

    # grav_var_h should be unchanged by the stencil
    assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)

    expected_grav_var_np = (
        expected_grav_var_h_np[:, :, :-1] + expected_grav_var_h_np[:, :, 1:]
    ) / 2
    assert np.array_equal(grav_var.field[:], expected_grav_var_np)


def test_compute_geopotential() -> None:
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend = "numpy"

    grav_var_h_np = np.random.random((nx, ny, nz + 1))
    expected_grav_var_h_np = copy.deepcopy(grav_var_h_np)
    grav_var_h = Quantity(
        data=grav_var_h_np,
        dims=example_dims,
        units="grav_var_h units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    gz_np = np.random.random((nx, ny, nz))
    gz_np_copy = copy.deepcopy(gz_np)
    gz = Quantity(
        data=gz_np,
        dims=example_dims,
        units="gz units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    zh_np = np.random.random((nx, ny, nz))
    expected_zh_np = copy.deepcopy(zh_np)
    zh = Quantity(
        data=zh_np,
        dims=example_dims,
        units="zh units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    compute_geopotential_np = stencil(
        backend=example_backend, definition=compute_geopotential
    )
    compute_geopotential_np(zh, gz, grav_var_h)

    # Check that zh and grav_var_h are unchanged
    assert np.array_equal(zh.field[:], expected_zh_np)
    assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)

    # Check that gz = zh * grav_var_h
    assert not np.array_equal(gz.field[:], gz_np_copy)
    # JK TODO: Is the expected_gz_np calculated correctly? double check with fortran or frank...
    expected_gz_np = zh_np * grav_var_h_np[:, :, :-1]
    assert np.array_equal(gz.field[:], expected_gz_np)


def setup_acoustic_dynamics(npx, npy, n_halo) -> Tuple[AcousticDynamics, DycoreState]:
    backend = "numpy"
    config = DynamicalCoreConfig(
        layout=(1, 1),
        npx=npx,
        npy=npy,
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
    dace_config = DaceConfig(communicator=communicator, backend=backend)
    stencil_config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend, rebuild=False, validate_args=True
        ),
        dace_config=dace_config,
    )
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=config.npx - 1,
        ny_tile=config.npy - 1,
        nz=config.npz,
        n_halo=n_halo,
        layout=config.layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, comm=communicator
    )
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    eta_file = Path(__file__).resolve().parents[1] / "data" / "eta79.nc"
    metric_terms = MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
        eta_file=eta_file,
    )
    grid_data = GridData.new_from_metric_terms(metric_terms)

    # create an initial state from the Jablonowski & Williamson Baroclinic
    # test case perturbation. JRMS2006
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
    stencil_factory = StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )

    dycore = DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        quantity_factory=quantity_factory,
        damping_coefficients=DampingCoefficients.new_from_metric_terms(metric_terms),
        config=config,
        timestep=timedelta(seconds=config.dt_atmos),
        phis=state.phis,
        state=state,
    )

    # JK TODO simplify this please, if possible...
    return dycore.acoustic_dynamics, state


# def test_acoustic_dynamics_init_average_gravity() -> None:
#     # Check that average gravity is called/used in AcousticDynamics initialization

#     nx = 12
#     ny = 12
#     nz = 79
#     n_halo = 3

#     # JK TODO: Why does the config need npx = nx-(2*n_halo)+1
#     ac_dyn, _ = setup_acoustic_dynamics(
#         nx - (2 * n_halo) + 1, ny - (2 * n_halo) + 1, n_halo
#     )  # nz is hard-coded to 79

#     # JK TODO: switch from example grav_var, grav_var_h to state.grav_var, state.grav_var_h

#     example_dims = ["I", "J", "K"]
#     example_backend = "numpy"

#     grav_var = Quantity(
#         data=np.zeros((nx, ny, nz)),
#         dims=example_dims,
#         units="grav_var units",
#         number_of_halo_points=n_halo,
#         backend=example_backend,
#     )

#     grav_var_h_np = np.random.random((nx, ny, nz + 1))
#     expected_grav_var_h_np = copy.deepcopy(grav_var_h_np)
#     grav_var_h = Quantity(
#         data=grav_var_h_np,
#         dims=example_dims,
#         units="grav_var_h units",
#         number_of_halo_points=n_halo,
#         backend=example_backend,
#     )

#     # Call ad_dyn._average_gravity. This is what we're testing.
#     ac_dyn._average_gravity(grav_var, grav_var_h)

#     # grav_var_h should be unchanged by the stencil
#     assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)

#     expected_grav_var_np = (
#         expected_grav_var_h_np[:, :, :-1] + expected_grav_var_h_np[:, :, 1:]
#     ) / 2
#     assert np.array_equal(grav_var.field[:], expected_grav_var_np)


# def test_acoustic_dynamics_call_average_gravity() -> None:
#     # Check that average gravity is called/used in AcousticDynamics call
#     nx = 12
#     ny = 12
#     nz = 79
#     n_halo = 3
#     timestep = 225  # JK TODO: Is this right?

#     # JK TODO: Why does the config need npx = nx-(2*n_halo)+1
#     ac_dyn, state = setup_acoustic_dynamics(
#         nx - (2 * n_halo) + 1, ny - (2 * n_halo) + 1, n_halo
#     )  # nz is hard-coded to 79

#     init_grav_var_np = copy.deepcopy(state.grav_var.field)
#     init_grav_var_h_np = copy.deepcopy(state.grav_var_h.field)

#     ac_dyn(state, timestep)

#     # The state.grav_var_h should be unchanged by the stencil.
#     assert np.array_equal(state.grav_var_h.field[:], init_grav_var_h_np)

#     # Check that the state.grav_var values match expectation:
#     expected_grav_var_np = (
#         init_grav_var_h_np[:, :, :-1] + init_grav_var_h_np[:, :, 1:]
#     ) / 2
#     assert np.array_equal(state.grav_var.field[:], expected_grav_var_np)


# """
# E        +  where False = <function array_equal at 0x7fd9f38750b0>(

# array([
# [[0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.],\n ...\n        [0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.]]]),

# array([[[0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.],\n ...\n        [0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.],\n        [0., 0., 0., ..., 0., 0., 0.]]]))

# E        +    where <function array_equal at 0x7fd9f38750b0> = np.array_equal

# """

# ############################ fv_dynamics.py


def test_init_gravity() -> None:
    # Check that init_gravity sets 3d grav_var to the constant GRAV for all vals
    backend = "numpy"
    nx_tile, ny_tile, nz, n_halo = 6, 6, 2, 3
    layout = (1, 1)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    mpi_comm = NullComm(rank=0, total_ranks=6, fill_value=0.0)
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    compilation_config = CompilationConfig(
        backend=backend, rebuild=False, validate_args=True
    )
    dace_config = DaceConfig(communicator=communicator, backend=backend)
    stencil_config = StencilConfig(
        compilation_config=compilation_config, dace_config=dace_config
    )
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx_tile,
        ny_tile=ny_tile,
        nz=nz,
        n_halo=n_halo,
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, comm=communicator
    )
    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    init_gravity_stencil = stencil_factory.from_origin_domain(
        set_value,
        origin=grid_indexing.origin_full(),
        domain=grid_indexing.domain_full(add=(0, 0, 1)),
    )
    grav_var: Quantity = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM],
        units="test",
        dtype=Float,
    )
    grav_var_h: Quantity = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_INTERFACE_DIM],
        units="test",
        dtype=Float,
    )
    init_gravity_stencil(grav_var, GRAV)
    init_gravity_stencil(grav_var_h, GRAV)
    assert np.all(grav_var.field == GRAV)
    assert np.all(grav_var_h.field == GRAV)
    # JK TODO: There's so much setup... Find a simpler way to set of stencil and quantity?


def test_adjust_gravity() -> None:
    # Check that adjust_gravity sets grav_var and grav_var_h are set appropriately
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend = "numpy"

    grav_var = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="grav_var units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    grav_var_h = Quantity(
        data=np.zeros((nx, ny, nz + 1)),
        dims=example_dims,
        units="grav_var_h units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    delz = Quantity(
        data=np.ones((nx, ny, nz)),
        dims=example_dims,
        units="delz units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    phis = Quantity(
        data=np.ones((nx, ny)),
        dims=["I", "J"],
        units="phis units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    adjust_gravity_numpy = stencil(backend=example_backend, definition=adjust_gravity)
    adjust_gravity_numpy(grav_var, grav_var_h, phis, delz)

    # Check that phis and delz are unchanged
    assert np.array_equal(phis.field[:], np.ones((nx, ny)))
    assert np.array_equal(delz.field[:], np.ones((nx, ny, nz)))

    # Check that grav_var and grav_var_h are set appropriately
    newrad = np.zeros((nx, ny))
    expected_grav_var_np = np.zeros((nx, ny, nz))
    expected_grav_var_h_np = np.zeros((nx, ny, nz + 1))

    assert np.array_equal(grav_var_h.field[:].shape, expected_grav_var_h_np.shape)

    for j in range(ny):
        for i in range(nx):
            for k in range(nz, -1, -1):
                if k == nz:
                    newrad[i, j] = RADIUS + (phis.field[i, j] / GRAV)
                else:
                    newrad[i, j] = newrad[i, j] - delz.field[i, j, k]
                expected_grav_var_h_np[i, j, k] = GRAV * (
                    (RADIUS**2) / (newrad[i, j] ** 2)
                )
                if k < nz:
                    expected_grav_var_np[i, j, k] = 0.5 * (
                        expected_grav_var_h_np[i, j, k + 1]
                        + expected_grav_var_h_np[i, j, k]
                    )

    # JK TODO: is there some rtol/atol threshold? the np vs stencil calculations are close but not exact.
    assert np.allclose(grav_var_h.field[:], expected_grav_var_h_np)
    assert np.allclose(grav_var.field[:], expected_grav_var_np)


# TODO JK NOTE to self --- checkout log_on_rank_0 for values that might be useful for test (possibly)
