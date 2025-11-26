import numpy as np
from typing import Any, Dict

from ndsl import StencilFactory, orchestrate
from f90nml import Namelist
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from pyFV3.stencils import DivergenceDamping
from pyFV3.testing import TranslateDycoreFortranData2Py
from pyFV3.utils.functional_validation import get_subset_func


class A2B_Ord4Compute:
    def __init__(self, stencil_factory: StencilFactory) -> None:
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            dace_compiletime_args=["divdamp"],
        )

    def __call__(
        self,
        divdamp,
        wk,
        vort,
        delpc,
        dt,
        grid_type: int,  # TODO: swap grid_type type hint when refactor into an enum
    ):
        # this function is kept because it has a translate test, if its
        # structure is changed significantly from __call__ of DivergenceDamping
        # consider deleting this method and the translate test, or altering the
        # savepoint to be more closely wrapped around a newly defined
        # gtscript function
        if divdamp._dddmp < 1e-5:
            divdamp._set_value(vort, 0.0)
        else:
            # TODO: what is wk/vort here?
            if grid_type < 3:
                divdamp.a2b_ord4(wk, vort)
                divdamp._smagorinksy_diffusion_approx_stencil(
                    delpc,
                    vort,
                    abs(dt),
                )
            else:
                pass


class TranslateA2B_Ord4(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        assert namelist.grid_type < 3
        self.in_vars["data_vars"] = {"wk": {}, "vort": {}, "delpc": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dt"]
        self.out_vars: Dict[str, Any] = {"wk": {}, "vort": {}}
        self.namelist = namelist  # type: ignore
        self.stencil_factory = stencil_factory
        self.compute_obj = A2B_Ord4Compute(stencil_factory)
        self._subset = get_subset_func(
            self.grid.grid_indexing,
            dims=[X_DIM, Y_DIM, Z_DIM],
            n_halo=((3, 3), (3, 3)),
        )

    def compute_from_storage(self, inputs):
        nord_col = self.grid.quantity_factory.zeros(dims=[Z_DIM], units="unknown")
        nord_col.data[:] = nord_col.np.asarray(inputs.pop("nord_col"))
        divdamp = DivergenceDamping(
            self.stencil_factory,
            self.grid.quantity_factory,
            self.grid.grid_data,
            self.grid.damping_coefficients,
            self.grid.nested,
            self.grid.stretched_grid,
            self.namelist.dddmp,
            self.namelist.d4_bg,
            self.namelist.nord,
            self.namelist.grid_type,
            nord_col,
            nord_col,
        )
        # TODO: use proper grid_type values when refactor into an enum
        inputs["grid_type"] = 0
        self.compute_obj(divdamp, **inputs)
        return inputs

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if varname in ["wk"]:
            return self._subset(output)
        else:
            return output
