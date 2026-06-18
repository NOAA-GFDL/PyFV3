from typing import Any, Dict

from f90nml import Namelist

from ndsl import StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyfv3.stencils import AdjustNegativeTracerMixingRatio
from pyfv3.testing import TranslateDycoreFortranData2Py
from pyfv3.tracers import FVTracers, FVTracersAxisName, default_GEOS_tracers


class TranslateNeg_Adj3(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "pt": {},
            "delp": {},
            "delz": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
        }
        self.in_vars["parameters"] = []
        self.out_vars: Dict[str, Any] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            # "pt": {},
        }
        self.stencil_factory = stencil_factory
        default_GEOS_tracers(grid.quantity_factory)

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_fn = AdjustNegativeTracerMixingRatio(
            self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            check_negative=self.config.check_negative,
            hydrostatic=self.config.hydrostatic,
        )
        tracers = self.grid.quantity_factory.empty(
            [I_DIM, J_DIM, K_DIM, FVTracersAxisName], "n/a"
        )
        tracers[:, :, :, FVTracers.index("vapor")] = inputs["qvapor"]
        tracers[:, :, :, FVTracers.index("liquid")] = inputs["qliquid"]
        tracers[:, :, :, FVTracers.index("rain")] = inputs["qrain"]
        tracers[:, :, :, FVTracers.index("snow")] = inputs["qsnow"]
        tracers[:, :, :, FVTracers.index("ice")] = inputs["qice"]
        tracers[:, :, :, FVTracers.index("graupel")] = inputs["qgraupel"]
        tracers[:, :, :, FVTracers.index("cloud")] = inputs["qcld"]

        compute_fn(
            tracers,
            inputs["pt"],
            inputs["delp"],
        )

        inputs["qvapor"] = tracers[:, :, :, FVTracers.index("vapor")]
        inputs["qliquid"] = tracers[:, :, :, FVTracers.index("liquid")]
        inputs["qrain"] = tracers[:, :, :, FVTracers.index("rain")]
        inputs["qsnow"] = tracers[:, :, :, FVTracers.index("snow")]
        inputs["qice"] = tracers[:, :, :, FVTracers.index("ice")]
        inputs["qgraupel"] = tracers[:, :, :, FVTracers.index("graupel")]
        inputs["qcld"] = tracers[:, :, :, FVTracers.index("cloud")]

        return self.slice_output(inputs)
