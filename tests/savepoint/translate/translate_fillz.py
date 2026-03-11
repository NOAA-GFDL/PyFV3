import numpy as np
from f90nml import Namelist

from ndsl import StencilFactory
from ndsl.stencils.testing import pad_field_in_j
from pyfv3.stencils import fillz
from pyfv3.testing import TranslateDycoreFortranData2Py
from pyfv3.tracers import setup_tracers


class TranslateFillz(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "dp2": {"istart": grid.is_, "iend": grid.ie, "axis": 1},
            "q2tracers": {"istart": grid.is_, "iend": grid.ie, "axis": 1},
        }
        self.in_vars["parameters"] = ["nq"]
        self.out_vars = {
            "q2tracers": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "axis": 1,
            }
        }
        self.max_error = 1e-13
        self.ignore_near_zero_errors = {"q2tracers": True}
        self.stencil_factory = stencil_factory
        self._quantity_factory = grid.quantity_factory

    def make_storage_data_input_vars(self, inputs, tracers):
        storage_vars = self.storage_vars()
        info = storage_vars["dp2"]
        inputs["dp2"] = self.make_storage_data(
            np.squeeze(inputs["dp2"]), istart=info["istart"], axis=info["axis"]
        )
        info = storage_vars["q2tracers"]
        tracers.quantity.field[:, :, :, :] = inputs["q2tracers"][:, np.newaxis, :, :]
        del inputs["q2tracers"]

    def compute(self, inputs):
        tracers = setup_tracers(
            number_of_tracers=inputs["q2tracers"].shape[2],
            quantity_factory=self._quantity_factory,
        )

        self.make_storage_data_input_vars(inputs, tracers)
        inputs["tracers"] = tracers

        for name, value in tuple(inputs.items()):
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        inputs.pop("nq")
        fillz_ = fillz.FillNegativeTracerValues(
            self.stencil_factory,
            self.grid.quantity_factory,
        )
        fillz_(**inputs)
        ds = self.grid.default_domain_dict()
        ds.update(self.out_vars["q2tracers"])
        out = {"q2tracers": tracers.quantity.field[:, 0, :, :]}
        return out
