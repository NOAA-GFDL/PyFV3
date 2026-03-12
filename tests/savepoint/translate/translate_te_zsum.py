from ndsl.stencils.testing import TranslateFortranData2Py
from pyfv3.stencils import moist_cv


class TranslateTe_Zsum(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.in_vars["data_vars"] = {
            "delp": {
                "kend": grid.npz,
            },
            "te_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "te0_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "zsum1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
        }
        self.out_vars = {
            "te_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "zsum1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
        }

        self.compute_func = stencil_factory.from_origin_domain(
            moist_cv.te_zsum,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

    def compute_from_storage(self, inputs):

        self.compute_func(
            inputs["te_2d_"],
            inputs["te0_2d_"],
            inputs["delp"],
            inputs["pkz"],
            inputs["zsum1"],
        )

        return inputs
