from ndsl import StencilFactory, Namelist
from ndsl.stencils.testing.grid import Grid
from ndsl.stencils.testing import TranslateFortranData2Py, ParallelTranslate
import numpy as np
from ndsl.typing import Communicator
from ndsl.quantity import Quantity
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

        # self.in_vars["data_vars"] = {
        #     "inputArray" :{
        #         "istart": grid.is_,
        #         "iend": grid.ie,
        #         "jstart": grid.js,
        #         "jend": grid.je,
        #     }
        # }
        # # self.in_vars["parameters"] = ["tesum"]
        # self.out_vars = {
        #     "inputArray" :{
        #         "istart": grid.is_,
        #         "iend": grid.ie,
        #         "jstart": grid.js,
        #         "jend": grid.je,
        #     }
        # }

        self._base.in_vars["data_vars"] = {
            "inputArray" :{
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "tesum":{

            }
        }

        self._base.out_vars = {
            "tesum":{

            }
        }

        self.NUMINT = 6
        self.NUMBIT = 46
        r_prec = 2.0 ** self.NUMBIT
        self.prec = 2 ** self.NUMBIT
        self.I_prec = 1.0 / (2.0 ** self.NUMBIT)
        self.pr = [r_prec**2, r_prec, 1.0, 1.0/r_prec, 1.0/r_prec**2, 1.0/r_prec**3]
        self.I_pr = [1.0/r_prec**2, 1.0/r_prec, 1.0, r_prec, r_prec**2, r_prec**3 ]
        self.prec_error = (2**62 + (2**62 - 1)) / 6
        

    def compute_from_storage(self, inputs):

        print('shape of inputArray = ', inputs["inputArray"].shape)
        print('prec_error = ', self.prec_error)
        print('inputArray = ', inputs["inputArray"])
        # print('type(inputs["inputArray"]) = ', type(inputs["inputArray"]))
        # self.array_manipulation(inputs["inputArray"][3:27,3:27])
        return inputs
    
    def compute_parallel(self, inputs, communicator: Communicator):
        # print("Communicator rank = ", communicator.rank)
        # print("Communicator size = ", communicator.size)
        # print('inputArray = ', inputs["inputArray"])
        # print('shape of inputArray = ', inputs["inputArray"].shape)
        # print("rank ", communicator.rank, ": sum(inputARray) = ", sum(sum(inputs["inputArray"])))
        # print('prec_error = ', self.prec_error)

        # print("tesum from translate test 1 : ", inputs["tesum"], type(inputs["tesum"]))

        inputs["tesum"] = self.mpp_global_sum(inputs["inputArray"], communicator, self.stencil_factory)

        # print("tesum from translate test 2 : ", inputs["tesum"])

        return inputs
    
    def mpp_global_sum(self, inputArray, communicator, stencil_factory=None):

        # print("rank ", communicator.rank, "sum(inputArray) = ", sum(sum(inputArray[0:24,0:24])))
        mag_max_term = 0.0
        # ints_sum = np.zeros((self.NUMINT))
        ints_sum = Quantity(
                            data=np.zeros((self.NUMINT),dtype=np.float64),
                            dims=["K"],
                            units="dunno",
                            gt4py_backend=stencil_factory.backend,
                        )
        
        ints_sum_reduce = Quantity(
                            data=np.zeros((self.NUMINT),dtype=np.float64),
                            dims=["K"],
                            units="dunno",
                            gt4py_backend=stencil_factory.backend,
                        )
        for j in range(0,24):
            for i in range(0,24):
                self.increment_ints_faster(ints_sum.data, inputArray[i,j], mag_max_term)

        # print("rank ", communicator.rank, "ints_sum = ", sum(ints_sum.data), ' before carry_over')
        self.carry_overflow(ints_sum.data, self.prec_error)
        # print("rank ", communicator.rank, "ints_sum = ", sum(ints_sum.data), ' after carry_over')

        communicator.all_reduce_sum(ints_sum, ints_sum_reduce)

        # print("rank ", communicator.rank, "sum(ints_sum_reduce) = ", sum(ints_sum_reduce.data), ' after all_reduce')
        self.regularize_ints(ints_sum_reduce.data)
        # print("rank ", communicator.rank,"ints_sum_reduce = ", sum(ints_sum_reduce.data), ' after regularize_ints')

        sum_ = self.ints_to_real(ints_sum_reduce.data)

        return sum_

        
    def increment_ints_faster(self, int_sum, r, max_mag_term):
        if((r >= 1e30) == r < 1e30):
            print("NaN_error")
            return
        sgn = 1
        if(r < 0.0):
            sgn = -1

        rs = abs(r)
        if(rs > abs(max_mag_term)):
            max_mag_term = r

        for i in range(0,self.NUMINT):
            ival = int(rs*self.I_pr[i])
            rs = rs - ival*self.pr[i]
            int_sum[i] = int_sum[i] + sgn*ival
    
    def carry_overflow(self, int_sum, prec_error):
        for i in range(self.NUMINT-1,1,-1):
            if abs(int_sum[i]) > self.prec:
                num_carry = int(int_sum[i] * self.I_prec)
                int_sum[i] = int_sum[i] - num_carry*self.prec
                int_sum[i-1] = int_sum[i-1] + num_carry
        if abs(int_sum[0]) > self.prec_error:
            overflow_error = True
    
    def regularize_ints(self, int_sum):
        for i in range(self.NUMINT-1, 0, -1):
            if abs(int_sum[i]) > self.prec:
                num_carry = int(int_sum[i] * self.I_prec)
                int_sum[i] = int_sum[i] - num_carry*self.prec
                int_sum[i-1] = int_sum[i-1] + num_carry
        
        positive = True

        for i in range(self.NUMINT):
            if abs(int_sum[i]) > 0:
                if int_sum[i] < 0:
                    positive = False   
                    break
        
        if positive:
            for i in range(self.NUMINT-1, 1, -1):
                if int_sum[i] < 0:
                    int_sum[i] = int_sum[i] + self.prec
                    int_sum[i-1] = int_sum[i-1] - 1
        
        else:
            for i in range(self.NUMINT-1, 1, -1):
                if int_sum[i] > 0:
                    int_sum[i] = int_sum[i] - self.prec
                    int_sum[i-1] = int_sum[i-1] + 1

    def ints_to_real(self, ints):
        r = 0.0

        for i in range(self.NUMINT):
            r = r + self.pr[i]*ints[i]

        return r