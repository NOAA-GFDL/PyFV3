from ndsl.quantity import Quantity
from ndsl.comm.comm_abc import ReductionOperator
import numpy as np

def mpp_global_sum(inputArray, communicator, stencil_factory=None):
    
    NUMINT = 6
    NUMBIT = 46
    r_prec = 2.0 ** NUMBIT
    prec = 2 ** NUMBIT
    I_prec = 1.0 / (2.0 ** NUMBIT)
    pr = [r_prec**2, r_prec, 1.0, 1.0/r_prec, 1.0/r_prec**2, 1.0/r_prec**3]
    I_pr = [1.0/r_prec**2, 1.0/r_prec, 1.0, r_prec, r_prec**2, r_prec**3 ]
    prec_error = (2**62 + (2**62 - 1)) / 6
    mag_max_term = 0.0

    ints_sum = Quantity(
                        data=np.zeros((NUMINT),dtype=np.float64),
                        dims=["K"],
                        units="dunno",
                        gt4py_backend=stencil_factory.backend,
                    )
    
    ints_sum_reduce = Quantity(
                        data=np.zeros((NUMINT),dtype=np.float64),
                        dims=["K"],
                        units="dunno",
                        gt4py_backend=stencil_factory.backend,
                    )
    
    # Note: This loop range in i and j are for the TBC test case.
    for j in range(inputArray.shape[1]):
        for i in range(inputArray.shape[0]):
            increment_ints_faster(ints_sum.data, pr, I_pr, inputArray[i,j], mag_max_term)

    # print("rank ", communicator.rank, "ints_sum = ", sum(ints_sum.data), ' before carry_over')
    carry_overflow(ints_sum.data, prec, I_prec, prec_error)
    # print("rank ", communicator.rank, "ints_sum = ", sum(ints_sum.data), ' after carry_over')

    communicator.all_reduce(ints_sum, ReductionOperator.SUM, ints_sum_reduce)

    # print("rank ", communicator.rank, "sum(ints_sum_reduce) = ", sum(ints_sum_reduce.data), ' after all_reduce')
    regularize_ints(ints_sum_reduce.data, prec, I_prec)
    # print("rank ", communicator.rank,"ints_sum_reduce = ", sum(ints_sum_reduce.data), ' after regularize_ints')

    sum_ = ints_to_real(ints_sum_reduce.data, pr)

    return sum_

    
def increment_ints_faster(int_sum, pr, I_pr, r, max_mag_term):
    if((r >= 1e30) == r < 1e30):
        print("NaN_error")
        return
    sgn = 1
    if(r < 0.0):
        sgn = -1

    rs = abs(r)
    if(rs > abs(max_mag_term)):
        max_mag_term = r

    for i in range(len(I_pr)):
        ival = int(rs*I_pr[i])
        rs = rs - ival*pr[i]
        int_sum[i] = int_sum[i] + sgn*ival

def carry_overflow(int_sum, prec, I_prec, prec_error):
    for i in range(len(int_sum)-1,1,-1):
        if abs(int_sum[i]) > prec:
            num_carry = int(int_sum[i] * I_prec)
            int_sum[i] = int_sum[i] - num_carry*prec
            int_sum[i-1] = int_sum[i-1] + num_carry
    if abs(int_sum[0]) > prec_error:
        overflow_error = True

def regularize_ints(int_sum, prec, I_prec):
    for i in range(len(int_sum)-1, 0, -1):
        if abs(int_sum[i]) > prec:
            num_carry = int(int_sum[i] * I_prec)
            int_sum[i] = int_sum[i] - num_carry*prec
            int_sum[i-1] = int_sum[i-1] + num_carry
    
    positive = True

    for i in range(len(int_sum)):
        if abs(int_sum[i]) > 0:
            if int_sum[i] < 0:
                positive = False   
                break
    
    if positive:
        for i in range(len(int_sum)-1, 1, -1):
            if int_sum[i] < 0:
                int_sum[i] = int_sum[i] + prec
                int_sum[i-1] = int_sum[i-1] - 1
    
    else:
        for i in range(len(int_sum)-1, 1, -1):
            if int_sum[i] > 0:
                int_sum[i] = int_sum[i] - prec
                int_sum[i-1] = int_sum[i-1] + 1

def ints_to_real(ints, pr):
    r = 0.0

    for i in range(len(ints)):
        r = r + pr[i]*ints[i]

    return r