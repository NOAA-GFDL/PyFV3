from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl import Namelist, StencilFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import FloatField
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from pyFV3.stencils.map_single import MapSingle


def rescale_delz_1(
    delz: FloatField,
    delp: FloatField,
):
    with computation(PARALLEL), interval(...):
        delz = -delz / delp


def rescale_delz_2(
    delz: FloatField,
    dp: FloatField,
):
    with computation(PARALLEL), interval(...):
        delz = -delz * dp


class TranslateMap1_PPM_delz(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "delz_": {
                "kend": grid.npz - 1,
            },
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "dp2_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "gz_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "delp": {},
        }
        self.in_vars["parameters"] = [
            "kord_wz",
        ]

        self.out_vars = {
            "delz_": {
                "kend": grid.npz - 1,
            },
        }

        # mode / iv set to 1 from GEOS
        self.mode = 1

        self.dims = [X_DIM, Y_DIM, Z_DIM]

        self._rescale_delz_1 = stencil_factory.from_origin_domain(
            rescale_delz_1,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

        self._rescale_delz_2 = stencil_factory.from_origin_domain(
            rescale_delz_2,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

    def compute_from_storage(self, inputs):
        self._compute_func = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_wz"],
            self.mode,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._rescale_delz_1(
            inputs["delz_"],
            inputs["delp"],
        )

        self._compute_func(
            inputs["delz_"],
            inputs["pe1_"],
            inputs["pe2_"],
            qs=inputs["gz_"],
        )

        self._rescale_delz_2(
            inputs["delz_"],
            inputs["dp2_3d"],
        )
        return inputs
