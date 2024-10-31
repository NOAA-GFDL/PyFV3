from ndsl import StencilFactory, Namelist
from ndsl.stencils.testing.grid import Grid
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.map_single import MapSingle

class TranslateMap1_PPM(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "w_": {
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
            "ws_":{
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },

        }
        self.in_vars["parameters"] = [
            "kord_wz",
        ]

        self.out_vars = {
            "w_": {
                "kend": grid.npz-1,
                },
            
        }

        # mode / iv set to -2 from GEOS
        self.mode = -2

        self.dims=[X_DIM, Y_DIM, Z_DIM]

    def compute_from_storage(self, inputs):
        self._compute_func = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_wz"],
            self.mode,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._compute_func(
                inputs["w_"],
                inputs["pe1_"],
                inputs["pe2_"],
                qs=inputs["ws_"],
                interp=False,
            )
        return inputs
