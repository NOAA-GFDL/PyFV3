from ndsl.dsl.typing import Float
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils import moist_cv


class TranslateMoistCVPlusPt_2d_last_step(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.in_vars["data_vars"] = {
            "qvapor": {
                "kend": grid.npz - 1,
            },
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
            "pt": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
        }

        self.in_vars["parameters"] = ["r_vir", "dtmp"]
        self.out_vars = {
            "pt": {},
        }

        self.compute_func = stencil_factory.from_origin_domain(
            moist_cv.moist_pt_last_step,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self.quantity_factory = grid.quantity_factory

        self._gz = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz,
            ),
            dtype=Float,
        )

    def compute_from_storage(self, inputs):

        self.compute_func(
            inputs["qvapor"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
            # self._gz,
            inputs["pt"],
            inputs["pkz"],
            Float(inputs["dtmp"]),
            inputs["r_vir"],
        )
        return inputs
