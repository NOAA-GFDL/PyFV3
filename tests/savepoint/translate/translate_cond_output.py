from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils import moist_cv


class TranslateCond_output(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.in_vars["data_vars"] = {
            "qliquid": {
                "kend": grid.npz - 1,
            },
            "qice": {
                "kend": grid.npz - 1,
            },
            "qrain": {
                "kend": grid.npz - 1,
            },
            "qsnow": {
                "kend": grid.npz - 1,
            },
            "qgraupel": {
                "kend": grid.npz - 1,
            },
            "q_con": {
                "kend": grid.npz - 1,
            },
        }

        self.out_vars = {
            "q_con": {
                "kend": grid.npz - 1,
            }
        }

        self.compute_func = stencil_factory.from_origin_domain(
            moist_cv.cond_output,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

    def compute_from_storage(self, inputs):

        self.compute_func(
            inputs["q_con"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
        )
        return inputs
