from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl import StencilFactory
from ndsl.dsl.typing import FloatField, Float
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.remap_profile import RemapProfile


class TranslateScalar_Profile(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "qs_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
                },
            "q4_2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_3": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_4": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "dp1_":{
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            

        }
        self.in_vars["parameters"] = [
            "q_min",
        ]

        self.out_vars = {
            "q4_1": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_2": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_3": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_4": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
        }

        # Value from GEOS
        self.kord = 9 

        # mode / iv set to 1 from GEOS
        self.mode = 1 

        self._compute_func = RemapProfile(
            self.stencil_factory,
            self.quantity_factory,
            self.kord,
            self.mode,
            dims=[X_DIM, Y_DIM, Z_DIM]
        )

    def compute_from_storage(self, inputs):
        self._compute_func(
                inputs["qs_"],
                inputs["q4_1"],
                inputs["q4_2"],
                inputs["q4_3"],
                inputs["q4_4"],
                inputs["dp1_"],
                Float(inputs["q_min"]),
            )
        return inputs
