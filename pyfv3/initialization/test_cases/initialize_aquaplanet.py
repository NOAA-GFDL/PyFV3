import numpy as np

import ndsl.constants as constants
import ndsl.dsl.gt4py_utils as utils
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.dsl.typing import Float
from ndsl.grid import GridData
from ndsl.grid.gnomonic import great_circle_distance_lon_lat, lon_lat_midpoint
from pyfv3.dycore_state import DycoreState
from pyfv3.initialization import init_utils


SURFACE_PRESSURE = Float(1.0e5)  # units of (Pa), from Table VI of DCMIP2016
NHALO = constants.N_HALO_DEFAULT


def aquaplanet_initialization(
    qvapor,
    qliquid,
    qice,
    qrain,
    qsnow,
    qgraupel,
    qo3mr,
    qsgs_tke,
    qcld,
    q_con,
    delp,
    u,
    v,
    pt,
    phis,
    ps,
    delz,
    w,
    ak,
    bk,
    hydrostatic,
    npz,
    nx,
    ny,
    area,
    hybrid_z,
):
    """
    Initializes state values for a cold-started aquaplanet simulation
    """
    alpha = 0
    
    # Initialize dry atmosphere:
    u[:] = 0.0
    v[:] = 0.0
    if not hydrostatic:
        w[:] = 0.0
    # TODO: Once tracers are handled appropriately set all to 0 here without hardcoding
    qvapor[:] = 0.0
    qliquid[:] = 0.0
    qice[:] = 0.0
    qrain[:] = 0.0
    qsnow[:] = 0.0
    qgraupel[:] = 0.0
    qo3mr[:] = 0.0
    qsgs_tke[:] = 0.0
    qcld[:] = 0.0
    q_con[:] = 0.0

    # Aqua-planet case: mean SLP=1.E5
    phis = 0.0
    
    init_utils.hydro_eq(npz, nx, ny, ps, phis, 1.0e5, delp, ak, bk, pt, delz, area, False, hydrostatic, hybrid_z)


def init_spherical_aquaplanet_state(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: CubedSphereCommunicator,
    is_steady: bool = False,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to cold start an
    aquaplanet.
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    nx, ny, nz = init_utils.local_compute_size(shape)
    numpy_state = init_utils.empty_numpy_dycore_state(shape)
    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:NHALO, :NHALO] = 0.0
    numpy_state.delp[:NHALO, NHALO + ny :] = 0.0
    numpy_state.delp[NHALO + nx :, :NHALO] = 0.0
    numpy_state.delp[NHALO + nx :, NHALO + ny :] = 0.0
    numpy_state.pe[:] = 0.0
    numpy_state.pt[:] = 1.0
    numpy_state.ua[:] = 1e35
    numpy_state.va[:] = 1e35
    numpy_state.uc[:] = 1e30
    numpy_state.vc[:] = 1e30
    numpy_state.w[:] = 1.0e30
    numpy_state.delz[:] = 1.0e25
    numpy_state.phis[:] = 1.0e25
    numpy_state.ps[:] = SURFACE_PRESSURE
    eta = np.zeros(nz)
    eta_v = np.zeros(nz)
    islice, jslice, slice_3d, slice_2d = init_utils.compute_slices(nx, ny)
    # Slices with extra buffer points in the horizontal dimension
    # to accomodate averaging over shifted calculations on the grid
    _, _, slice_3d_buffer, slice_2d_buffer = init_utils.compute_slices(nx + 1, ny + 1)

    aquaplanet_initialization(
        eta=eta,
        eta_v=eta_v,
        delp=numpy_state.delp[slice_3d],
        ps=numpy_state.ps[slice_2d],
        pe=numpy_state.pe[slice_3d],
        ak=utils.asarray(grid_data.ak.data),
        bk=utils.asarray(grid_data.bk.data),
    )
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)

    comm.vector_halo_update(state.u, state.v, n_points=NHALO)

    return state
