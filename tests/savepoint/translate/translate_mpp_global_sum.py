from ndsl import StencilFactory
from f90nml import Namelist
from ndsl.stencils.testing import ParallelTranslate
from ndsl.stencils.testing.grid import Grid
from ndsl.typing import Communicator
from pyFV3.mpi.mpp_sum import MPPGlobalSum


class TranslateMpp_global_sum(ParallelTranslate):
    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid

        self._base.in_vars["data_vars"] = {
            "inputArray": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "tesum": {},
        }

        self._base.out_vars = {"tesum": {}}

    def compute_parallel(self, inputs, communicator: Communicator):
        mpp_sum = MPPGlobalSum(self.stencil_factory, communicator)
        inputs["tesum"] = mpp_sum(inputs["inputArray"])

        return inputs
