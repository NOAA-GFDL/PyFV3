from ndsl import Namelist, StencilFactory
from ndsl.stencils.testing import ParallelTranslate
from ndsl.stencils.testing.grid import Grid
from ndsl.typing import Communicator
from pyFV3.stencils.mpp_global_sum import mpp_global_sum


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

        inputs["tesum"] = mpp_global_sum(
            inputs["inputArray"],
            communicator,
            self.stencil_factory,
        )

        return inputs
