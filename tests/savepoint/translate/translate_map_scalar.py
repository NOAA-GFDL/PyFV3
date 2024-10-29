from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl import StencilFactory
from ndsl.dsl.typing import FloatField, Float
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.map_single import MapSingle


class TranslateMap_Scalar(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "q1": {
                "kend": grid.npz-1,
                },
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },

        }
        self.in_vars["parameters"] = [
            "q_min",
        ]

        self.out_vars = {
            "q1": {
                "kend": grid.npz-1,
                },
            
        }

        # Value from GEOS
        self._kord_tm = 9 

        # mode / iv set to 1 from GEOS
        self.mode = 1 

        self.dims=[X_DIM, Y_DIM, Z_DIM]

        self._compute_func = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            self._kord_tm,
            self.mode,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def compute_from_storage(self, inputs):
        self._compute_func(
                inputs["q1"],
                inputs["pe1_"],
                inputs["pe2_"],
                qmin=Float(inputs["q_min"]),
                interp=True,
            )
        return inputs
