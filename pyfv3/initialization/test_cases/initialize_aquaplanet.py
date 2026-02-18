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


def init_aquaplanet_state(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    hydrostatic: bool,
    comm: CubedSphereCommunicator,
) -> DycoreState:
    sample_quantity = grid_data.lat
    field_shape = (*sample_quantity.field.shape[0:2], grid_data.ak.data.shape[0])
    data_shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    nx, ny, nz = init_utils.local_compute_size(data_shape)
    numpy_state = init_utils.empty_numpy_dycore_state(data_shape)
    isc, iec, jsc, jec = init_utils.local_compute_bounds(field_shape)
    print(isc, iec, jsc, jec)
    print(nx, ny, nz)

    hybrid_z = False

    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:NHALO, :NHALO] = 0.0
    numpy_state.delp[:NHALO, NHALO + ny:] = 0.0
    numpy_state.delp[NHALO + nx:, :NHALO] = 0.0
    numpy_state.delp[NHALO + nx:, NHALO + ny:] = 0.0
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
    _, _, slice_3d_buffer, slice_2d_buffer = init_utils.compute_slices(
        nx + 1,
        ny + 1
    )

    init_utils.setup_pressure_fields(
        eta=eta,
        eta_v=eta_v,
        delp=numpy_state.delp[slice_3d],
        ps=numpy_state.ps[slice_2d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pk=numpy_state.pk[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ak=utils.asarray(grid_data.ak.data),
        bk=utils.asarray(grid_data.bk.data),
        ptop=grid_data.ptop,
    )
    alpha = 0
    # Initialize dry atmosphere
    numpy_state.qvapor[:] = 3.e-6
    numpy_state.qliquid[:] = 3.e-6
    numpy_state.qice[:] = 3.e-6
    numpy_state.qrain[:] = 3.e-6
    numpy_state.qsnow[:] = 3.e-6
    numpy_state.qgraupel[:] = 3.e-6
    numpy_state.qo3mr[:] = 3.e-6
    numpy_state.qsgs_tke[:] = 3.e-6
    numpy_state.qcld[:] = 3.e-6
    numpy_state.q_con[:] = 3.e-6
    numpy_state.u[:] = 0.0
    numpy_state.v[:] = 0.0
    if not hydrostatic: 
        numpy_state.w[:] = 0.0

    numpy_state.phis[:] = 0.0
    print(numpy_state.ps.shape)
    init_utils.hydro_eq(
        nz, isc, iec, jsc, jec, numpy_state.ps[:], numpy_state.phis[:], 1.e5,
        numpy_state.delp[:], grid_data.ak.data[:], grid_data.bk.data[:],
        numpy_state.pt[:], numpy_state.delz[:], grid_data.area.data[:],
        NHALO, False, hydrostatic, hybrid_z, comm)

    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)

    comm.vector_halo_update(state.u, state.v, n_points=NHALO)

    return state
