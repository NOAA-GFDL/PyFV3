from ndsl import Quantity, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, FloatFieldK
from ndsl.stencils import corners


def double_copy(q_in: FloatField, copy_1: FloatField, copy_2: FloatField):
    with computation(PARALLEL), interval(...):
        copy_1 = q_in
        copy_2 = q_in


def copy(q_in: FloatField, q_copy: FloatField):
    with computation(PARALLEL), interval(...):
        q_copy = q_in


def compute_weighted_average(
    dp_ref: FloatFieldK,
    vel: FloatField,
    avg: FloatField,
):
    """
    Perform a cubic spline interpolation of wind velocity from grid center to grid edge

    Args:
        dp_ref(in): layer thickness in Pa
        vel(in): grid center wind speed
        avg(out: interpolated (grid edge) wind speed
    """
    # there's some complexity due to gz being defined on interfaces
    # have to interpolate winds to layer interfaces first, using higher-order
    with computation(PARALLEL):
        with interval(0, 1):
            top_ratio = dp_ref / (dp_ref + dp_ref[1])
            avg = vel + (vel - vel[0, 0, 1]) * top_ratio
        with interval(1, -1):
            int_ratio = 1.0 / (dp_ref[-1] + dp_ref)
            avg = (dp_ref * vel[0, 0, -1] + dp_ref[-1] * vel) * int_ratio
        with interval(-1, None):
            bot_ratio = dp_ref[-1] / (dp_ref[-2] + dp_ref[-1])
            avg = vel[0, 0, -1] + (vel[0, 0, -1] - vel[0, 0, -2]) * bot_ratio


def compute_fx_fy(
    gz_x: FloatField,
    gz_y: FloatField,
    xfx: FloatField,
    yfx: FloatField,
    fx: FloatField,
    fy: FloatField,
):
    """
    Compute first-order upwind fluxes of gz in x and y directions.

    Args:
        gz_x(in): gz with corners copied to perform derivatives in x-direction
        gz_y(in): gz with corners copied to perform derivatives in y-direction
        xfx(in): contravariant c-grid u-wind interpolated to layer interfaces,
            including metric terms to make it a "volume flux"
        yfx(in): contravariant c-grid v-wind interpolated to layer interfaces
        fx(out): first-order upwind x-flux of gz
        fy(out): first-order upwind y-flux of gz
    """

    with computation(PARALLEL), interval(...):
        if xfx > 0.0:
            fx = gz_x[-1, 0, 0]
        else:
            fx = gz_x
        fx = xfx * fx

        if yfx > 0.0:
            fy = gz_y[0, -1, 0]
        else:
            fy = gz_y
        fy = yfx * fy


def compute_gz_ws(
    gz_y: FloatField,
    area: FloatFieldIJ,
    fx: FloatField,
    fy: FloatField,
    xfx: FloatField,
    yfx: FloatField,
    dz_min: Float,
    dt: Float,
    zs: FloatFieldIJ,
    ws: FloatFieldIJ,
    gz: FloatField,
):
    """
        Compute gz and wd, eusures gz is monotonically increasing in z at the end

    Args
        gz_y(in): gz with corners copied to perform derivatives in y-direction
        area(in):
        fx(in): first-order upwind x-flux of gz
        fy(in): first-order upwind y-flux of gz
        xfx(in): contravariant c-grid u-wind interpolated to layer interfaces,
            including metric terms to make it a "volume flux"
        yfx(in): contravariant c-grid v-wind interpolated to layer interfaces
        dz_min(in): Controls minimum thickness in NH solver
        dt(in): timestep over which to evolve the geopotential height, in seconds
        zs(in): surface height in m
        ws(out): lagrangian (parcel-following) surface vertical wind implied by
            lowest-level gz change note that a parcel moving horizontally
            across terrain will be moving in the vertical (eqn 5.5 in documentation)
        gz(out): geopotential height on model interfaces
    """

    with computation(PARALLEL), interval(...):
        gz = (gz_y * area + (fx - fx[1, 0, 0]) + (fy - fy[0, 1, 0])) / (
            area + (xfx - xfx[1, 0, 0]) + (yfx - yfx[0, 1, 0])
        )
    with computation(FORWARD), interval(...):
        rdt = 1.0 / dt
        ws = (zs - gz) * rdt
    with computation(BACKWARD), interval(0, -1):
        gz_kp1 = gz[0, 0, 1] + dz_min
        gz = gz if gz > gz_kp1 else gz_kp1


class UpdateGeopotentialHeightOnCGrid:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        area: Quantity,
        dp_ref: Quantity,
        grid_type: int,
        dz_min: Float,
    ):
        """
        Args:
            dz_min: controls minimum thickness in NH solver
        """

        grid_indexing = stencil_factory.grid_indexing
        self._area = area
        self._grid_type = grid_type
        self._dz_min = dz_min
        # TODO: this is needed because GridData.dp_ref does not have access
        # to a QuantityFactory, we should add a way to perform operations on
        # Quantity and persist the QuantityFactory choices
        # e.g. by adding a quantity.factory
        # attribute, or by implementing basic math like slicing, addition, etc.
        # here it's needed to ensure we have a buffer point after the compute domain
        self._dp_ref = quantity_factory.zeros(
            dp_ref.dims,
            units=dp_ref.units,
            dtype=Float,
        )
        self._dp_ref.view[:] = dp_ref.view[:]
        self._gz_x = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="m**2/s**2",
            dtype=Float,
        )
        self._gz_y = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="m**2/s**2",
            dtype=Float,
        )
        self._gz_filled = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="m**2/s**2",
            dtype=Float,
        )
        self._xfx = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._yfx = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._fx = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        self._fy = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=Float,
        )
        full_origin = grid_indexing.origin_full()
        full_domain = grid_indexing.domain_full(add=(0, 0, 1))
        self._double_copy_stencil = stencil_factory.from_origin_domain(
            double_copy,
            origin=full_origin,
            domain=full_domain,
        )
        self._copy_stencil = stencil_factory.from_origin_domain(
            copy,
            origin=full_origin,
            domain=full_domain,
        )

        ax_offsets = grid_indexing.axis_offsets(full_origin, full_domain)

        if self._grid_type < 3:
            self._fill_corners_x_stencil = stencil_factory.from_origin_domain(
                corners.fill_corners_2cells_x_stencil,
                externals=ax_offsets,
                origin=full_origin,
                domain=full_domain,
            )
            self._fill_corners_y_stencil = stencil_factory.from_origin_domain(
                corners.fill_corners_2cells_y_stencil,
                externals=ax_offsets,
                origin=full_origin,
                domain=full_domain,
            )

        self._compute_weighted_average = stencil_factory.from_origin_domain(
            compute_weighted_average,
            origin=grid_indexing.origin_compute(add=(-1, -1, 0)),
            domain=grid_indexing.domain_compute(add=(3, 3, 1)),
        )

        self._compute_flux = stencil_factory.from_origin_domain(
            compute_fx_fy,
            origin=grid_indexing.origin_compute(add=(-1, -1, 0)),
            domain=grid_indexing.domain_compute(add=(3, 3, 1)),
        )

        self._compute_gz_ws = stencil_factory.from_origin_domain(
            compute_gz_ws,
            origin=grid_indexing.origin_compute(add=(-1, -1, 0)),
            domain=grid_indexing.domain_compute(add=(2, 2, 1)),
        )

        self.DEBUG_VAR_1 = quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "n/a")

    def __call__(
        self,
        zs: FloatFieldIJ,
        ut: FloatField,
        vt: FloatField,
        gz: FloatField,
        ws: FloatFieldIJ,
        dt: Float,
    ):
        """
        Step dz forward on c-grid

        Args:
            dp_ref: layer thickness in Pa
            zs: surface height in m
            ut: horizontal wind (TODO: covariant or contravariant?)
            vt: horizontal wind (TODO: covariant or contravariant?)
            gz: geopotential height on model interfaces
            ws: surface vertical wind implied by horizontal motion over topography
            dt: timestep over which to evolve the geopotential height, in seconds
        """

        # TODO: is this advecting gz, and if so can we name it that?
        # Can we reduce duplication of advection logic with other stencils?

        self._double_copy_stencil(gz, self._gz_x, self._gz_y)

        if self._grid_type < 3:
            self._fill_corners_x_stencil(self._gz_x, self._gz_x)
            self._fill_corners_y_stencil(self._gz_y, self._gz_y)

        self._compute_weighted_average(dp_ref=self._dp_ref, vel=ut, avg=self._xfx)
        self._compute_weighted_average(dp_ref=self._dp_ref, vel=vt, avg=self._yfx)

        self._compute_flux(
            gz_x=self._gz_x,
            gz_y=self._gz_y,
            xfx=self._xfx,
            yfx=self._yfx,
            fx=self._fx,
            fy=self._fy,
        )

        self._compute_gz_ws(
            gz_y=self._gz_y,
            area=self._area,
            fx=self._fx,
            fy=self._fy,
            xfx=self._xfx,
            yfx=self._yfx,
            dz_min=self._dz_min,
            dt=dt,
            zs=zs,
            ws=ws,
            gz=gz,
        )
