from ndsl.stencils.testing import TranslateFortranData2Py, pad_field_in_j
from pyFV3.stencils import moist_cv

class TranslateMoistCVPlusTe_2d(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "qvapor_js"},
            "qliquid": {"serialname": "qliquid_js"},
            "qice": {"serialname": "qice_js"},
            "qrain": {"serialname": "qrain_js"},
            "qsnow": {"serialname": "qsnow_js"},
            "qgraupel": {"serialname": "qgraupel_js"},
            "delp": {},
            "pt": {},
            "phis_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
            },
            "te_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "u": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz,
            },
            "v": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz,
            },
            "w":{
                "kend": grid.npz,
            },
            "cosa_s": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "rsin2": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
        }
        self.write_vars = ["qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel"]
        for k, v in self.in_vars["data_vars"].items():
            # if k not in self.write_vars:
            if k in self.write_vars:
                v["axis"] = 1

        self.out_vars = {
            "te_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
        }

        self.compute_func = stencil_factory.from_origin_domain(
            moist_cv.moist_te,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

    def compute_from_storage(self, inputs):
        for name, value in inputs.items():
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )

        self.compute_func(inputs["qvapor"],
                          inputs["qliquid"],
                          inputs["qrain"],
                          inputs["qsnow"],
                          inputs["qice"],
                          inputs["qgraupel"],
                          inputs["u"],
                          inputs["v"],
                          inputs["w"],
                          inputs["te_2d_"],
                          inputs["pt"],
                          inputs["phis_"],
                          inputs["delp"],
                          inputs["rsin2"],
                          inputs["cosa_s"],
                        )

        return inputs
