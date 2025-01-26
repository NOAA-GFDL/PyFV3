from typing import List
import numpy as np

import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
    i32,
)

from ndsl import (
    QuantityFactory,
    StencilFactory,
    WrappedHaloUpdater,
    orchestrate,
    Quantity,
)
from ndsl.grid import GridData
from ndsl.constants import (
    N_HALO_DEFAULT,
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
)
from ndsl.dsl.typing import FloatField, FloatFieldIJ, FloatFieldK
from ndsl.comm.communicator import Communicator, ReductionOperator
from pyFV3.stencils.fvtp2d import FiniteVolumeTransport
from pyFV3.tracers import Tracers
from ndsl.dsl.gt4py_utils import asarray
from ndsl.utils import safe_assign_array


@gtscript.function
def flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
        xfx = (
            cx * dxa[-1, 0] * dy * sin_sg3[-1, 0] if cx > 0 else cx * dxa * dy * sin_sg1
        )
    return xfx


@gtscript.function
def flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
        yfx = (
            cy * dya[0, -1] * dx * sin_sg4[0, -1] if cy > 0 else cy * dya * dx * sin_sg2
        )
    return yfx


def flux_compute(
    cx: FloatField,
    cy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    xfx: FloatField,
    yfx: FloatField,
):
    """
    Args:
        cx (in):
        cy (in):
        dxa (in):
        dya (in):
        dx (in):
        dy (in):
        sin_sg1 (in):
        sin_sg2 (in):
        sin_sg3 (in):
        sin_sg4 (in):
        xfx (out): x-direction area flux
        yfx (out): y-direction area flux
    """
    with computation(PARALLEL), interval(...):
        xfx = flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx)
        yfx = flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx)


def divide_fluxes_by_n_substeps(
    cxd: FloatField,
    xfx: FloatField,
    mfxd: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    mfyd: FloatField,
    cmax: FloatFieldK,
):
    """
    Divide all inputs in-place by the number of substeps n_split computed
    from the max courant number on the grid

    Args:
        cxd (inout):
        xfx (inout):
        mfxd (inout):
        cyd (inout):
        yfx (inout):
        mfyd (inout):
    """
    with computation(PARALLEL), interval(...):
        n_split = i32(1.0 + cmax)
        if n_split > 1:
            frac = 1.0 / n_split
            cxd = cxd * frac
            xfx = xfx * frac
            mfxd = mfxd * frac
            cyd = cyd * frac
            yfx = yfx * frac
            mfyd = mfyd * frac


def apply_mass_flux(
    dp1: FloatField,
    x_mass_flux: FloatField,
    y_mass_flux: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    """
    Args:
        dp1 (in): initial pressure thickness
        mfx (in): flux of (area * mass * g) in x-direction
        mfy (in): flux of (area * mass * g) in y-direction
        rarea (in): 1 / area
        dp2 (out): final pressure thickness
    """
    with computation(PARALLEL), interval(...):
        dp2 = (
            dp1
            + (
                (x_mass_flux - x_mass_flux[1, 0, 0])
                + (y_mass_flux - y_mass_flux[0, 1, 0])
            )
            * rarea
        )


def apply_tracer_flux(
    q: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
    cmax: FloatFieldK,
    current_nsplit: int,
):
    """
    Args:
        q (inout):
        dp1 (in):
        fx (in):
        fy (in):
        rarea (in):
        dp2 (in):
    """
    with computation(PARALLEL), interval(...):
        if current_nsplit < i32(1.0 + cmax):
            q = (q * dp1 + ((fx - fx[1, 0, 0]) + (fy - fy[0, 1, 0])) * rarea) / dp2


# Simple stencil replacing:
#   self._tmp_dp2[:] = dp1
#   dp1[:] = dp2
#   dp2[:] = self._tmp_dp2
# Because dpX can be a quantity or an array
def swap_dp(dp1: FloatField, dp2: FloatField):
    with computation(PARALLEL), interval(...):
        tmp = dp1
        dp1 = dp2
        dp2 = tmp


class TracerAdvection:
    """
    Performs horizontal advection on tracers.

    Corresponds to tracer_2D_1L in the Fortran code.

    Args:
        stencil_factory: Stencil maker built on the required grid
        quantity_factory: Quantity maker built on the required grid
        transport: The Finite Volume to be applied to each tracers
        grid_data: Metric Terms for the grid
        comm: Communicator on the grid
        tracers: Bundle of data of tracers to be advected
        exclude_tracers: Tracers to not be advected
        update_mass_courant: update the mass and courant numbers
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        transport: FiniteVolumeTransport,
        grid_data: GridData,
        comm: Communicator,
        tracers: Tracers,
        exclude_tracers: List[str],
        update_mass_courant: bool = True,
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["tracers"],
        )
        grid_indexing = stencil_factory.grid_indexing
        self.grid_indexing = grid_indexing  # needed for selective validation
        self.grid_data = grid_data
        self._exclude_tracers = exclude_tracers
        self._update_mass_courant = update_mass_courant

        if not self._update_mass_courant:
            self._tmp_mfx = quantity_factory.zeros(
                [X_INTERFACE_DIM, Y_DIM, Z_DIM],
                units="unknown",
            )
            self._tmp_mfy = quantity_factory.zeros(
                [X_DIM, Y_INTERFACE_DIM, Z_DIM],
                units="unknown",
            )
            self._tmp_cx = quantity_factory.zeros(
                [X_INTERFACE_DIM, Y_DIM, Z_DIM],
                units="unknown",
            )
            self._tmp_cy = quantity_factory.zeros(
                [X_DIM, Y_INTERFACE_DIM, Z_DIM],
                units="unknown",
            )

        self._x_area_flux = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            units="unknown",
        )
        self._y_area_flux = quantity_factory.zeros(
            [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            units="unknown",
        )
        self._x_flux = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            units="unknown",
        )
        self._y_flux = quantity_factory.zeros(
            [X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
            units="unknown",
        )
        self._tmp_dp = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Pa",
        )
        self._tmp_dp2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="Pa",
        )
        self._cmax = quantity_factory.zeros(
            [Z_DIM],
            units="unitless",
        )

        ax_offsets = grid_indexing.axis_offsets(
            grid_indexing.origin_full(), grid_indexing.domain_full()
        )
        local_axis_offsets = {}
        for axis_offset_name, axis_offset_value in ax_offsets.items():
            if "local" in axis_offset_name:
                local_axis_offsets[axis_offset_name] = axis_offset_value

        self._swap_dp = stencil_factory.from_origin_domain(
            swap_dp,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )

        self._flux_compute = stencil_factory.from_origin_domain(
            flux_compute,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._divide_fluxes_by_n_substeps = stencil_factory.from_origin_domain(
            divide_fluxes_by_n_substeps,
            origin=grid_indexing.origin_full(),
            domain=grid_indexing.domain_full(add=(1, 1, 0)),
            externals=local_axis_offsets,
        )
        self._apply_mass_flux = stencil_factory.from_origin_domain(
            apply_mass_flux,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self._apply_tracer_flux = stencil_factory.from_origin_domain(
            apply_tracer_flux,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
            externals=local_axis_offsets,
        )
        self.finite_volume_transport: FiniteVolumeTransport = transport

        # Setup halo updater for tracers
        tracer_halo_spec = quantity_factory.get_quantity_halo_spec(
            dims=[X_DIM, Y_DIM, Z_DIM],
            n_halo=N_HALO_DEFAULT,
        )

        # We can exclude tracers from advecting and therefore also
        # halo exchanging
        advected_tracers = {}
        for name, tracer in tracers.items():
            if name in exclude_tracers:
                continue
            advected_tracers[name] = tracer
        self._tracers_halo_updater = WrappedHaloUpdater(
            comm.get_scalar_halo_updater([tracer_halo_spec] * len(advected_tracers)),
            advected_tracers,
            [t for t in advected_tracers.keys()],
        )

        # Setup tracer courant max reduction calculation
        self._compute_cmax = TracerCMax(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            grid_data=grid_data,
            comm=comm,
        )

    def __call__(
        self,
        tracers: Tracers,
        dp1,
        x_mass_flux,
        y_mass_flux,
        x_courant,
        y_courant,
    ):
        """
        Apply advection to tracers based on the given courant numbers and mass fluxes.

        Note only output values for tracers are used, all other inouts are only such
        because they are modified for intermediate computation.

        Args:
            tracers (inout): tracers to advect according to fluxes during
                acoustic substeps
            dp1 (in): pressure thickness of atmospheric layers before acoustic substeps
            x_mass_flux (inout): total mass flux in x-direction over acoustic substeps
            y_mass_flux (inout): total mass flux in y-direction over acoustic substeps
            x_courant (inout): accumulated courant number in x-direction
            y_courant (inout): accumulated courant number in y-direction
        """

        if self._update_mass_courant:
            working_x_mass_flux = x_mass_flux
            working_y_mass_flux = x_mass_flux
            working_x_courant = x_courant
            working_y_courant = y_courant
        else:
            safe_assign_array(self._tmp_mfx.data, x_mass_flux)
            safe_assign_array(self._tmp_mfy.data, y_mass_flux)
            safe_assign_array(self._tmp_cx.data, x_courant)
            safe_assign_array(self._tmp_cy.data, y_courant)
            working_x_mass_flux = self._tmp_mfx
            working_y_mass_flux = self._tmp_mfy
            working_x_courant = self._tmp_cx
            working_y_courant = self._tmp_cy

        self._flux_compute(
            working_x_courant,
            working_y_courant,
            self.grid_data.dxa,
            self.grid_data.dya,
            self.grid_data.dx,
            self.grid_data.dy,
            self.grid_data.sin_sg1,
            self.grid_data.sin_sg2,
            self.grid_data.sin_sg3,
            self.grid_data.sin_sg4,
            self._x_area_flux,
            self._y_area_flux,
        )

        self._compute_cmax(
            cx=working_x_courant,
            cy=working_y_courant,
            cmax=self._cmax,
        )

        self._divide_fluxes_by_n_substeps(
            working_x_courant,
            self._x_area_flux,
            working_x_mass_flux,
            working_y_courant,
            self._y_area_flux,
            working_y_mass_flux,
            self._cmax,
        )

        self._tracers_halo_updater.update()

        dp2 = self._tmp_dp

        # The original algorithm works on K level independantly
        # (from with a  K loop) and therefore compute `nsplit`
        # per K
        # The stencil nature of the framework doesn't allow for it
        # because after advection, an halo exchange need to be carried
        # (or else we could just move the test within the stencil).
        # We overcompute to retain true parallelization, by running
        # a loop on the highest number of nsplit, but restraining
        # actual update in `apply_tracer_flux` to only the valid
        # K level for each tracers
        cmax_on_host = asarray(self._cmax.view[:], to_type=np.ndarray)
        max_n_split = i32(1.0 + cmax_on_host.max())

        for current_nsplit in range(int(max_n_split)):
            last_call = current_nsplit == max_n_split - 1
            # tracer substep
            self._apply_mass_flux(
                dp1,
                working_x_mass_flux,
                working_y_mass_flux,
                self.grid_data.rarea,
                dp2,
            )
            for name, q in tracers.items():
                if name in self._exclude_tracers:
                    pass
                else:
                    self.finite_volume_transport(
                        q,
                        working_x_courant,
                        working_y_courant,
                        self._x_area_flux,
                        self._y_area_flux,
                        self._x_flux,
                        self._y_flux,
                        x_mass_flux=x_mass_flux,
                        y_mass_flux=y_mass_flux,
                    )
                    self._apply_tracer_flux(
                        q,
                        dp1,
                        self._x_flux,
                        self._y_flux,
                        self.grid_data.rarea,
                        dp2,
                        cmax=self._cmax,
                        current_nsplit=current_nsplit,
                    )
            if not last_call:
                self._tracers_halo_updater.update()
                # we can't use variable assignment to avoid a data copy
                # because of current dace limitations
                self._swap_dp(dp1, dp2)


def cmax_stencil_low_k(
    cx: FloatField,
    cy: FloatField,
    cmax: FloatField,
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


def cmax_stencil_high_k(
    cx: FloatField,
    cy: FloatField,
    sin_sg5: FloatFieldIJ,
    cmax: FloatField,
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


class TracerCMax:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        comm: Communicator,
    ):
        """Perform global courant number max.

        The maximum courant number for every atmospheric level on the entire grid.
        """
        self._grid_data = grid_data
        self._comm = comm
        grid_indexing = stencil_factory.grid_indexing
        cmax_atmospheric_level_split = int(grid_indexing.domain[2] / 6) - 1
        self._cmax_low_k = stencil_factory.from_origin_domain(
            func=cmax_stencil_low_k,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                cmax_atmospheric_level_split,
            ),
        )
        self._cmax_high_k = stencil_factory.from_origin_domain(
            func=cmax_stencil_high_k,
            origin=(
                grid_indexing.origin_compute()[0],
                grid_indexing.origin_compute()[1],
                cmax_atmospheric_level_split,
            ),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                grid_indexing.domain[2] - cmax_atmospheric_level_split,
            ),
        )
        self._tmp_cmax = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
        )
        self._tmp_cmax_in_K = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
        )

    def __call__(self, cx: Quantity, cy: Quantity, cmax: Quantity):
        if __debug__:
            if not isinstance(cmax, Quantity):
                raise TypeError(
                    f"[pyFV3][Tracer]: cmax must be a quantity, got {type(cmax)}"
                )
        self._cmax_low_k(
            cx=cx,
            cy=cy,
            cmax=self._tmp_cmax,
        )
        self._cmax_high_k(
            cx=cx,
            cy=cy,
            sin_sg5=self._grid_data.sin_sg5,
            cmax=self._tmp_cmax,
        )
        cmax.data[:] = self._tmp_cmax.data.max(axis=0).max(axis=0)[:]
        self._comm.all_reduce_per_element_in_place(cmax, ReductionOperator.MAX)
