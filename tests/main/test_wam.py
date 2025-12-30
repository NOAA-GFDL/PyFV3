from pathlib import Path
from dataclasses import field
import copy

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
from ndsl.grid import GridData, MetricTerms
from ndsl.dsl.typing import Float, FloatField
from ndsl.constants import GRAV, RADIUS, X_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from ndsl.dsl.gt4py import stencil
from pyfv3 import DynamicalCoreConfig, DycoreState
from pyfv3.initialization import init_utils
from pyfv3.initialization.analytic_init import AnalyticCase
from pyfv3.stencils.fv_dynamics import adjust_gravity, init_gravity, init_gravity_h
from pyfv3.stencils.dyn_core import average_gravity, compute_geopotential

# use numpy for now until I figure out how to use FloatField
import numpy as np # JK TODO: Should I be using xumpy?

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
    # JK TODO: Is this really useful as a test?


############################ dyn_core.py
def test_average_gravity() -> None:
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend="numpy"

    grav_var = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="grav_var units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    grav_var_h_np = np.random.random((nx, ny, nz+1))
    expected_grav_var_h_np = copy.deepcopy(grav_var_h_np)
    grav_var_h = Quantity(
        data=grav_var_h_np,
        dims=example_dims,
        units="grav_var_h units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )
    
    average_gravity_numpy = stencil(backend=example_backend, definition=average_gravity)
    average_gravity_numpy(grav_var, grav_var_h)

    # grav_var_h should be unchanged by the stencil
    assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)

    expected_grav_var_np = (expected_grav_var_h_np[:,:,:-1]+expected_grav_var_h_np[:,:,1:]) / 2
    assert np.array_equal(grav_var.field[:], expected_grav_var_np)


def test_compute_geopotential() -> None:
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend="numpy"

    grav_var_h_np = np.random.random((nx, ny, nz+1))
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

    compute_geopotential_np = stencil(backend=example_backend, definition=compute_geopotential)
    compute_geopotential_np(zh, gz, grav_var_h)

    # Check that zh and grav_var_h are unchanged
    assert np.array_equal(zh.field[:], expected_zh_np)
    assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)

    # Check that gz = zh * grav_var_h
    assert not np.array_equal(gz.field[:], gz_np_copy)
    # JK TODO: Is the expected_gz_np calculated correctly? double check with fortran or frank...
    expected_gz_np = zh_np * grav_var_h_np[:,:,:-1]
    assert np.array_equal(gz.field[:], expected_gz_np)


def test_p_grad_c_stencil() -> None:
    # Check that
    # 1. addition of average_gravity stencil and
    # 2. addition of grav_var_h parameter in self._compute_geopotential_stencil
    # still produces reasonable results
    
    assert False # TODO

############################ fv_dynamics.py

def test_init_gravity() -> None:
    # Check that init_gravity sets 3d grav_var to the constant GRAV for all vals
    backend = "numpy"
    nx_tile, ny_tile, nz, n_halo = 6, 6, 2, 3
    layout = (1,1)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    mpi_comm = NullComm(rank=0, total_ranks=6, fill_value=0.0)
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)
    compilation_config = CompilationConfig(backend=backend, rebuild=False, validate_args=True)
    dace_config = DaceConfig(communicator=communicator, backend=backend)
    stencil_config = StencilConfig(compilation_config=compilation_config, dace_config=dace_config)
    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=nx_tile,
        ny_tile=ny_tile,
        nz=nz,
        n_halo=n_halo,
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank
    )
    grid_indexing = GridIndexing.from_sizer_and_communicator(sizer=sizer, comm=communicator)
    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)
    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)
    init_gravity_stencil = stencil_factory.from_dims_halo(
        init_gravity,
        compute_dims=[X_DIM, Y_DIM, Z_DIM],
        compute_halos=(n_halo, n_halo),
    )
    grav_var: Quantity = quantity_factory.zeros(
        [X_DIM, Y_DIM, Z_DIM],
        units="test",
        dtype=Float,
    )
    init_gravity_stencil(grav_var)
    assert np.all(grav_var.field == GRAV)
    # JK TODO: There's so much setup... Find a simpler way to set of stencil and quantity?

def test_init_gravity_h() -> None:
    # Check that init_gravity sets 3d grav_var_h to constant GRAV for all vals
    # same as above? Why is init_gravity and init_gravity_h the same? maybe will be different in future?

    # Using 01_gt4py_basics.ipynb for a simpler approach than test_init_gravity()
    nx = 5
    ny = 5
    nz = 2
    shape = (nx, ny, nz+1)
    n_halos = 3

    example_data = np.zeros(shape)
    example_dims = ["I", "J", "K"]
    example_units = "test units"
    example_backend="numpy"

    example_qty = Quantity(
        data=example_data,
        dims=example_dims,
        units=example_units,
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    init_gravity_h_numpy = stencil(backend=example_backend, definition=init_gravity_h)
    init_gravity_h_numpy(example_qty)

    assert np.all(example_qty.field == GRAV)


def test_adjust_gravity() -> None:
    # Check that adjust_gravity sets grav_var and grav_var_h are set appropriately
    nx = 5
    ny = 5
    nz = 2
    n_halos = 3

    example_dims = ["I", "J", "K"]
    example_backend="numpy"

    grav_var = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="grav_var units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    grav_var_h = Quantity(
        data=np.zeros((nx, ny, nz+1)),
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

    # 2D quantities:
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
    expected_grav_var = Quantity(
        data=np.zeros((nx, ny, nz)),
        dims=example_dims,
        units="grav_var units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )
    expected_grav_var_h = Quantity(
        data=np.zeros((nx, ny, nz+1)),
        dims=example_dims,
        units="grav_var_h units",
        number_of_halo_points=n_halos,
        backend=example_backend,
    )

    # Calculated numpy versions of expected values


    # Look at Fortran version and figure out a better test.

    # ValueError: operands could not be broadcast together with shapes (5,5) (5,5,2)
    # newrad = RADIUS + (phis/GRAV) - delz

    # TODO: There's gotta be a better way to do this to avoid the valueerror:
    newrad = np.zeros((nx, ny))
    expected_grav_var_np = np.zeros((nx, ny, nz))
    expected_grav_var_h_np = np.zeros((nx, ny, nz+1))
    
    assert np.array_equal(grav_var_h.field[:].shape, expected_grav_var_h_np.shape)

    ##### Stencil:
    # with computation(FORWARD), interval(-1,None):
    #     newrad = RADIUS + (phis/GRAV)
    #     grav_var_h = GRAV*(RADIUS**2)/newrad**2
    ##### Fortran:
    #    do j=js,je
    #      do i=is,ie
    #        newrad(i,j) = radius + (phis(i,j)/grav)
    #        grav_var_h(i,j,npz+1) = grav*((radius**2)/(newrad(i,j)**2))
    #      enddo
    #    enddo

    #newrad[:,:] = RADIUS + (phis/GRAV)
    #expected_grav_var_h_np[:,:,-1] += GRAV*(RADIUS**2)/newrad**2

    # dumb iterative sanity check:
    for j in range(ny):
        for i in range(nx):
            newrad[i,j] = RADIUS + (phis.field[i,j]/GRAV)
            expected_grav_var_h_np[i,j,-1] += GRAV*(RADIUS**2)/newrad[i,j]**2

    ##### Stencil:
    # with computation(BACKWARD), interval(...):
    #    newrad = newrad - delz
    #    grav_var_h = GRAV*(RADIUS**2)/newrad**2
    #    grav_var = 0.5*(grav_var_h[0, 0, 1] + grav_var_h[0, 0, 0])
    ##### Fortran:
    #     do j=js,je
    #       do i=is,ie
    #        do k=npz,1,-1
    #          newrad(i,j) = newrad(i,j) - delz(i,j,k)
    #          grav_var_h(i,j,k) = grav*((radius**2)/(newrad(i,j)**2))
    #          grav_var(i,j,k) = 0.5*(grav_var_h(i,j,k+1)+grav_var_h(i,j,k))
    #        enddo
    #      enddo
    #    enddo

    for j in range(ny):
        for i in range(nx):
            for k in range(nz-1, 0, -1):
                newrad[i,j] = newrad[i,j] - delz.field[i,j,k]
                expected_grav_var_h_np[i,j,k] = GRAV*((RADIUS**2)/(newrad[i,j]**2))
                expected_grav_var_np[i,j,k] = 0.5*(expected_grav_var_h_np[i,j,k+1]+expected_grav_var_h_np[i,j,k])

    #k=1
    #newrad += -delz.field[:,:,k]
    #expected_grav_var_h_np[:,:,k] = GRAV*((RADIUS**2)/(newrad**2))
    #expected_grav_var_np[:,:,k] = 0.5*(expected_grav_var_h_np[:,:,k+1]+expected_grav_var_h_np[:,:,k])

    #k=0
    #newrad += -delz.field[:,:,k]
    #expected_grav_var_h_np[:,:,k] = GRAV*((RADIUS**2)/(newrad**2))
    #expected_grav_var_np[:,:,k] = 0.5*(expected_grav_var_h_np[:,:,k+1]+expected_grav_var_h_np[:,:,k])

    assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)
    assert np.array_equal(grav_var.field[:], expected_grav_var_np)
    

# TODO JK NOTE to self --- checkout log_on_rank_0 for values that might be useful for test (possibly)


"""
>       assert np.array_equal(grav_var_h.field[:], expected_grav_var_h_np)
E       assert False
E        +  where False = <function array_equal at 0x7f10cd303bb0>(

array([[[3.98073395e+14, 9.80665276e+00, 0.00000000e+00],\n        [0.00000000e+00, 9.80665276e+00, 0.00000000e+00],\n  ...,\n        [0.00000000e+00, 9.80665276e+00, 0.00000000e+00],\n        [3.98073395e+14, 9.80665276e+00, 0.00000000e+00]]]), 

array([[[0.        , 9.80665276, 9.80664969],\n        [0.        , 9.80665276, 9.80664969],\n        [0.        , 9.806... 9.80665276, 9.80664969],\n        [0.        , 9.80665276, 9.80664969],\n        [0.        , 9.80665276, 9.80664969]]]))
E        +    where <function array_equal at 0x7f10cd303bb0> = np.array_equal

tests/main/test_wam.py:378: AssertionError
=
"""
