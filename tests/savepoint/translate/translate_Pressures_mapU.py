from f90nml import Namelist
from pyFV3.stencils.map_single import MapSingle
from pyFV3.stencils.remapping import pe0_ptop_xmax, pressures_mapu

from ndsl import StencilFactory
from ndsl.constants import I_DIM, J_DIM, J_INTERFACE_DIM, K_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid


class TranslatePressures_mapU(TranslateFortranData2Py):
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
                "kend": grid.npz,
            },
            "ak": {},
            "bk": {},
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz,
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz,
            },
            "u_": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
                "kend": grid.npz - 1,
            },
            "mfy_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "cy_": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
        }
        self.in_vars["parameters"] = [
            "ptop",
            "kord_mt",
        ]

        self.out_vars = {
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz,
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz,
            },
            "u_": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
                "kend": grid.npz - 1,
            },
            "mfy_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "cy_": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
        }

        grid_indexing = stencil_factory.grid_indexing

        self.dims = [I_DIM, J_DIM, K_DIM]

        self._pressures_mapu = stencil_factory.from_origin_domain(
            pressures_mapu,
            origin=grid_indexing.origin_compute(),
            domain=(grid_indexing.domain[0], 1, grid_indexing.domain[2] + 1),
        )

        self._pe0_ptop_xmax = stencil_factory.from_origin_domain(
            pe0_ptop_xmax,
            origin=(grid_indexing.domain[0] + 3, 3, 0),
            domain=(1, 1, grid_indexing.domain[2] + 1),
        )

    def compute_from_storage(self, inputs):
        self._map1_ppm_u = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_mt"],
            -1,
            dims=[I_DIM, J_INTERFACE_DIM, K_DIM],
        )

        self._pressures_mapu(
            inputs["pe_"],
            inputs["ak"],
            inputs["bk"],
            inputs["pe0_"],
            inputs["pe3_"],
            inputs["ptop"],
        )

        self._pe0_ptop_xmax(
            inputs["pe0_"],
            inputs["ptop"],
        )

        self._map1_ppm_u(
            inputs["u_"],
            inputs["pe0_"],
            inputs["pe3_"],
        )
        self._map1_ppm_u(
            inputs["mfy_"],
            inputs["pe0_"],
            inputs["pe3_"],
        )

        self._map1_ppm_u(
            inputs["cy_"],
            inputs["pe0_"],
            inputs["pe3_"],
        )
        return inputs
