from f90nml import Namelist
from pyFV3.stencils.remap_profile import RemapProfile

from ndsl import StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid


class TranslateCS_Profile(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "qs_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_3": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_4": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "dp1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
        }
        self.in_vars["parameters"] = [
            "iv_",
            "kord_",
        ]

        self.out_vars = {
            "q4_1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_3": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "q4_4": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
        }

    def compute_from_storage(self, inputs):
        self._compute_func = RemapProfile(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_"],
            inputs["iv_"],
            dims=[I_DIM, J_DIM, K_DIM],
        )

        self._compute_func(
            inputs["qs_"],
            inputs["q4_1"],
            inputs["q4_2"],
            inputs["q4_3"],
            inputs["q4_4"],
            inputs["dp1_"],
        )
        return inputs
