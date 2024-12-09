from ndsl import StencilFactory, Namelist
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing import ParallelTranslate
from ndsl.typing import Communicator
from mpi4py import MPI

class TranslateGetMPIProp(ParallelTranslate):
    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        print("Base TranslateGetMPIProp is initialized")
        super().__init__(grid, namelist, stencil_factory)
        self._base.in_vars["data_vars"] = {
            "delz" :{

            }
        }
        self._base.out_vars = {
            "delz" :{
                
            }
        }

    def compute_parallel(self, inputs, communicator: Communicator):
        print("Communicator rank = ", communicator.rank)
        print("Communicator size = ", communicator.size)
        global_sum = communicator.comm.allreduce(communicator.rank, op=MPI.SUM)
        print("global_sum of ranks = ", global_sum)
        return inputs