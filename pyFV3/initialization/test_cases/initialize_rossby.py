""" Test case initialization for Rossby-Haurwitz wave 4

Corresponds to Fortran shallow-water test #6 found in tools/test_cases.F90 of
https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere.git 
"""

import math
import numpy as np

import ndsl.constants as constants
from ndsl import CubedSphereCommunicator, QuantityFactory
from ndsl.grid import GridData
from pyFV3.dycore_state import DycoreState
from pyFV3.initialization import init_utils


NHALO = constants.N_HALO_DEFAULT
OMG = 7.848e-6
RK = 7.848e-6
R = 4.0 # Wave Number (likely)
GH0 = 8.0e3 * constants.GRAV


def preinit_for_all_sw(state: DycoreState,
                       shape,
                       grid_data: GridData
):
    """ Pre-initialization for all shallow water tests

    Args:
        state: DycoreState modified to update pe, pt, delp
        shape: tuple
        grid_data: GridData
    """    
    state.pe[:] = 0.0
    state.pt[:] = 1.0

    # Initialize Halo Corners
    nx, ny, _ = init_utils.local_compute_size(shape)
    state.delp[:NHALO, :NHALO] = 0.0
    state.delp[:NHALO, NHALO + ny :] = 0.0
    state.delp[NHALO + nx :, :NHALO] = 0.0
    state.delp[NHALO + nx :, NHALO + ny :] = 0.0


def init_rossby_winds(p1, p2):
    """ Initializes u or v winds specific to Rossby-Haurwitz wave test

    Args
        p1 : np.ndarray
        p2 : np.ndarray

    Returns
        np.ndarray representing u or v D-winds
    """
    muv = init_utils._find_midpoint_unit_vectors(p1, p2)
    p3 = muv["midpoint"]
    e2 = muv["unit_dir"] 
    ex = muv["exv"] 
    ey = muv["eyv"]
    utmp = (
        constants.RADIUS * OMG * np.cos(p3[:, :, 1]) + constants.RADIUS * RK
        * (np.cos(p3[:, :, 1])**(R-1))
        * (R * np.sin(p3[:, :, 1])**2 - np.cos(p3[:, :, 1])**2)*np.cos(R*p3[:, :, 0])
    )
    vtmp = (
        -1 * constants.RADIUS * RK * R * np.sin(p3[:, :, 1])
        * np.sin(R * p3[:, :, 0]) * np.cos(p3[:, :, 1])**(R-1)
    )
    return utmp * np.sum(e2 * ex, 2) + vtmp * np.sum(e2 * ey, 2)


def init_rossby_delp(state: DycoreState,
                     grid_data: GridData
):
    """ Initializes delp specific to Rossby-Haurwitz wave test

    Args
        state DycoreState
        grid_Data GridData

    Returns
        np.ndarray representing delp values
    """
    agd0 = grid_data.lon_agrid.data[:]
    agd1 = grid_data.lat_agrid.data[:]    
    A = (0.5 * OMG * (2 * constants.OMEGA + OMG) * (np.cos(agd1)**2)
         + 0.25 * RK * RK * (np.cos(agd1)**(R + R))
         * ((R + 1) * (np.cos(agd1)**2) + (2 * R * R - R - 2) - 2 * (R * R) * np.cos(agd1)**(-2)))
    B = ((2 * (constants.OMEGA + OMG) * RK / ((R+1) * (R+2)))
         * (np.cos(agd1)**R) * ((R*R+2 * R + 2) - ((R + 1) * np.cos(agd1))**2 ))
    C = 0.25 * RK * RK * (np.cos(agd1)**(2 * R)) * ((R + 1) * (np.cos(agd1)**2) - (R+2))    
    return (GH0 + constants.RADIUS * constants.RADIUS
            * ( A + B * np.cos(R * agd0) + C * np.cos(2 * R * agd0)))


def init_for_rossby(state: DycoreState,
                    grid_data: GridData
):
    """ Initialization specific to Rossby-Haurwitz wave test

    Args
        state DycoreState, modified to update the 
        grid_Data GridData
    """
    
    state.phis[:] = 0.0

    # Initialize delp
    state.delp[:,:,0] = init_rossby_delp(state, grid_data)
    state.delp[:,:,0] = state.delp[:,:,0] - state.phis[:]

    grid = np.transpose(
        np.stack(
            [grid_data._horizontal_data.lon.data, grid_data._horizontal_data.lat.data]
        ),
        [1, 2, 0],
    )

    # Initialize u winds
    p1 = grid[:-1, :, :]
    p2 = grid[1:, :, :]
    state.u[:-1, :, 0] = init_rossby_winds(p1, p2)

    # Initialize v winds
    p1 = grid[:, :-1, :]
    p2 = grid[:, 1:, :]
    state.v[:, :-1, 0] = init_rossby_winds(p1, p2)
    
    # TODO: Pay attention to the slice indices. u and v are similarly calculated.

    """ From test_cases.F90 (case 6): 
         call mp_update_dwinds(u, v, npx, npy, npz, domain, bd)
         call dtoa( u, v,ua,va,dx,dy,dxa,dya,dxc,dyc,npx,npy,ng,bd)
         !call mpp_update_domains( ua, va, domain, gridtype=AGRID_PARAM)
         call atoc(ua,va,uc,vc,dx,dy,dxa,dya,npx,npy,ng, gridstruct%bounded_domain, domain, bd)
         initWindsCase=initWindsCase6
    """


def postinit_for_all_sw(state, grid_data):
    """ Post-initialization from test_cases.F90 that applies to all shallow water tests

    Args:
        state: DycoreState - modified
    """

    # NOTE: The cl/cl2 tracers from the original Fortran test_cases.F90 are not brought over.
    # NOTE: A-grid and C-grid winds are not initialized here.

    state.delp[:,:,1:] = state.delp[:,:,0][:,:,np.newaxis]
    state.u[:,:,1:] = state.u[:,:,0][:,:,np.newaxis]
    state.v[:,:,1:] = state.v[:,:,0][:,:,np.newaxis]
    state.ps[:] = state.delp[:,:,0]
    
    
def init_rossby_state(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    comm: CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState TODO: explain more

    Args:
        grid_data:              current selected grid data values
        quantity_factory:       QuantityFactory
        comm:                   CubedSphereCommunicator

    Returns:
        DycoreState
    """
    sample_quantity = grid_data.lat
    shape = (*sample_quantity.data.shape[0:2], grid_data.ak.data.shape[0])
    numpy_state = init_utils.empty_numpy_dycore_state(shape)

    preinit_for_all_sw(numpy_state, shape, grid_data)
    init_for_rossby(numpy_state, grid_data)
    postinit_for_all_sw(numpy_state, grid_data)

    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=NHALO)
    comm.vector_halo_update(state.u, state.v, n_points=NHALO)
    # TODO: anymore comm updates? delp?

    return state
