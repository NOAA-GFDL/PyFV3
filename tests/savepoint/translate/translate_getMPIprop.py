from ndsl import StencilFactory, Namelist
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing import ParallelTranslate
from ndsl.typing import Communicator
from ndsl.quantity import Quantity
import numpy as np

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

        len_k = 10

        a = [1,2,3,4,5]

        self._testQuantity_1D = Quantity(
                                        data=np.array(a,dtype=np.float32),
                                        dims=["K"],
                                        units="dunno",
                                        gt4py_backend=stencil_factory.backend,
                                    )
        
        self._testQuantity_2D = Quantity(
                                        data=np.ones([5,5],dtype=np.float32),
                                        dims=["I","J"],
                                        units="dunno2",
                                        gt4py_backend=stencil_factory.backend,
                                    )
        
        self._testQuantity_3D = Quantity(
                                        data=np.ones([3,3,3],dtype=np.float32),
                                        dims=["I","J","K"],
                                        units="dunno3",
                                        gt4py_backend=stencil_factory.backend,
                                    )
        
    def compute_parallel(self, inputs, communicator: Communicator):
        print("Communicator rank = ", communicator.rank)
        print("Communicator size = ", communicator.size)
        print("self._testQuantity = ", self._testQuantity_1D.data)
        global_sum_q = communicator.all_reduce_sum(self._testQuantity_1D)
        print("global_sum_q.data = ", global_sum_q.data)
        print("global_sum_q.metadata = ", global_sum_q.metadata)

        global_sum_q = communicator.all_reduce_sum(self._testQuantity_2D)
        print("global_sum_q.data = ", global_sum_q.data)
        print("global_sum_q.metadata = ", global_sum_q.metadata)

        global_sum_q = communicator.all_reduce_sum(self._testQuantity_3D)
        print("global_sum_q.data = ", global_sum_q.data)
        print("global_sum_q.metadata = ", global_sum_q.metadata)

        return inputs