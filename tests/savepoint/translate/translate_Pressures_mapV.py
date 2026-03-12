from f90nml import Namelist
from pyFV3.stencils.map_single import MapSingle
from pyFV3.stencils.remapping import pressures_mapv

from ndsl import StencilFactory
from ndsl.constants import I_DIM, I_INTERFACE_DIM, J_DIM, K_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid


class TranslatePressures_mapV(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "pe_": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "ak": {},
            "bk": {},
            "v_": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
            "mfx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "cx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
        }
        self.in_vars["parameters"] = [
            "kord_mt",
        ]

        self.out_vars = {
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "v_": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
            "mfx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "cx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
        }

        grid_indexing = stencil_factory.grid_indexing

        self.dims = [I_DIM, J_DIM, K_DIM]

        self._pressures_mapv = stencil_factory.from_origin_domain(
            pressures_mapv,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0] + 1,
                1,
                grid_indexing.domain[2] + 1,
            ),
        )

    def compute_from_storage(self, inputs):
        self._map1_ppm_v = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_mt"],
            -1,
            dims=[I_INTERFACE_DIM, J_DIM, K_DIM],
        )

        self._pressures_mapv(
            inputs["pe_"],
            inputs["ak"],
            inputs["bk"],
            inputs["pe0_"],
            inputs["pe3_"],
        )

        self._map1_ppm_v(
            inputs["v_"],
            inputs["pe0_"],
            inputs["pe3_"],
        )

        self._map1_ppm_v(
            inputs["mfx_"],
            inputs["pe0_"],
            inputs["pe3_"],
        )

        self._map1_ppm_v(
            inputs["cx_"],
            inputs["pe0_"],
            inputs["pe3_"],
        )
        return inputs
