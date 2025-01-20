import ndsl.dsl.gt4py_utils as utils
from ndsl import Namelist, StencilFactory
from ndsl.stencils.testing import TranslateGrid
from pyFV3.stencils import XPiecewiseParabolic
from pyFV3.testing import TranslateDycoreFortranData2Py


class TranslateXPPM(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "q": {"serialname": "xppm_q", "jstart": "jfirst"},
            "c": {"serialname": "xppm_c", "istart": grid.is_},
        }
        self.in_vars["parameters"] = ["iord", "jfirst", "jlast"]
        self.out_vars = {
            "xppm_flux": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": "jfirst",
                "jend": "jlast",
            }
        }
        self.stencil_factory = stencil_factory

    def jvars(self, inputs):
        inputs["jfirst"] += TranslateGrid.fpy_model_index_offset
        inputs["jlast"] += TranslateGrid.fpy_model_index_offset
        inputs["jfirst"] = self.grid.global_to_local_y(inputs["jfirst"])
        inputs["jlast"] = self.grid.global_to_local_y(inputs["jlast"])

    def process_inputs(self, inputs):
        self.jvars(inputs)
        self.make_storage_data_input_vars(inputs)

    def compute(self, inputs):
        self.process_inputs(inputs)
        inputs["xppm_flux"] = utils.make_storage_from_shape(
            inputs["q"].shape, backend=self.stencil_factory.backend
        )
        origin = self.grid.grid_indexing.origin_compute()
        domain = self.grid.grid_indexing.domain_compute(add=(1, 1, 0))
        self.compute_func = XPiecewiseParabolic(
            stencil_factory=self.stencil_factory,
            dxa=self.grid.dxa,
            grid_type=self.grid.grid_type,
            iord=int(inputs["iord"]),
            origin=(origin[0], int(inputs["jfirst"]), origin[2]),
            domain=(domain[0], int(inputs["jlast"] - inputs["jfirst"] + 1), domain[2]),
        )
        self.compute_func(inputs["q"], inputs["c"], inputs["xppm_flux"])
        return self.slice_output(inputs)


class TranslateXPPM_2(TranslateXPPM):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"]["q"]["serialname"] = "xppm_q2"
        self.out_vars["xppm_flux"]["serialname"] = "xppm_flux_2"
