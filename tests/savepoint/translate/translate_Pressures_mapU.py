from ndsl import StencilFactory, Namelist
from ndsl.stencils.testing.grid import Grid
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.remapping import pressures_mapu

class TranslatePressures_mapU(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "pe_": {
                "istart": grid.is_-1,
                "iend": grid.ie+1,
                "jstart": grid.js-1,
                "jend": grid.je+1,
                "kend": grid.npz
                },
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "ak":{

            },
            "bk":{

            }

        }
        self.in_vars["parameters"] = [
            "ptop",
        ]

        self.out_vars = {
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            
        }

        grid_indexing = stencil_factory.grid_indexing

        self.dims=[X_DIM, Y_DIM, Z_DIM]

        self._domain_jextra = (
            grid_indexing.domain[0],
            grid_indexing.domain[1] + 1,
            grid_indexing.domain[2] + 1,
        )

        self._compute_func = stencil_factory.from_origin_domain(
            pressures_mapu,
            origin=grid_indexing.origin_compute(),
            domain=self._domain_jextra,
        )

    def compute_from_storage(self, inputs):
        self._compute_func(
                inputs["pe_"],
                inputs["ak"],
                inputs["bk"],
                inputs["pe0_"],
                inputs["pe3_"],
                inputs["ptop"],
            )
        return inputs
