import numpy as np
from ndsl import StencilFactory
from f90nml import Namelist
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from pyFV3.stencils import NonhydrostaticVerticalSolverCGrid
from pyFV3.testing import TranslateDycoreFortranData2Py
from pyFV3.utils.functional_validation import get_subset_func


class TranslateRiem_Solver_C(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.compute_func = NonhydrostaticVerticalSolverCGrid(  # type: ignore
            stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            p_fac=namelist.p_fac,
        )
        self.in_vars["data_vars"] = {
            "cappa": {},
            "hs": {},
            "w3": {},
            "ptc": {},
            "q_con": {},
            "delpc": {},
            "gz": {},
            "pef": {},
            "ws": {},
        }
        self.in_vars["parameters"] = ["dt2", "ptop"]
        self.out_vars = {"pef": {"kend": grid.npz}, "gz": {"kend": grid.npz}}
        self.max_error = 5e-14
        self.stencil_factory = stencil_factory
        self._subset = get_subset_func(
            self.grid.grid_indexing,
            dims=[X_DIM, Y_DIM, Z_DIM],
            n_halo=((3, 3), (3, 3)),
        )

    def compute(self, inputs):
        outputs = super().compute(inputs)
        outputs["gz"] = self.subset_output("gz", outputs["gz"])
        return outputs

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if varname in ["gz", "pef"]:
            return self._subset(output)
        else:
            return output
