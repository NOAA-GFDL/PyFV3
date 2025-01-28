from ndsl.dsl.typing import Float
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from pyFV3.stencils.w_fix_consrv_moment import W_fix_consrv_moment


class TranslateW_fix_consrv_moment(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.compute_func = stencil_factory.from_origin_domain(
            func=W_fix_consrv_moment,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

        self.in_vars["data_vars"] = {
            "w": {
                "kend": grid.npz - 1,
            },
            "dp2_W": grid.compute_dict(),
        }

        self.in_vars["parameters"] = ["w_max", "w_min"]

        self.out_vars = {
            "w": {
                "kend": grid.npz - 1,
            },
        }
        self._gz = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=Float,
        )

        self._w2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz,
            ),
            dtype=Float,
        )

        self._compute_performed = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=bool,
        )

    def compute_from_storage(self, inputs):

        self.compute_func(
            inputs["w"],
            self._w2,
            inputs["dp2_W"],
            self._gz,
            inputs["w_max"],
            inputs["w_min"],
            self._compute_performed,
        )

        return inputs
