import warnings

import numpy as np

from ndsl import Quantity, StencilFactory
from ndsl.comm.communicator import Communicator, ReductionOperator
from ndsl.dsl.typing import Float


def _increment_ints_faster(
    int_sum: np.ndarray,
    pr: list[float],
    I_pr: list[float],
    r: float,
    max_mag_term: float,
) -> None:
    if (r >= 1e30) == r < 1e30:
        print("NaN_error")
        return
    sgn = 1
    if r < 0.0:
        sgn = -1

    rs = abs(r)
    if rs > abs(max_mag_term):
        max_mag_term = r

    for i in range(len(I_pr)):
        ival = int(rs * I_pr[i])
        rs = rs - ival * pr[i]
        int_sum[i] = int_sum[i] + sgn * ival


def _carry_overflow(
    int_sum: np.ndarray,
    prec: int,
    I_prec: float,
    prec_error: float,
) -> bool:
    overflow_error = False
    for i in range(len(int_sum) - 1, 0, -1):
        if abs(int_sum[i]) > prec:
            num_carry = int(int_sum[i] * I_prec)
            int_sum[i] = int_sum[i] - num_carry * prec
            int_sum[i - 1] = int_sum[i - 1] + num_carry
    if abs(int_sum[0]) > prec_error:
        overflow_error = True
    return overflow_error


def _regularize_ints(int_sum: np.ndarray, prec: int, I_prec: float) -> None:
    for i in range(len(int_sum) - 1, 0, -1):
        if abs(int_sum[i]) > prec:
            num_carry = int(int_sum[i] * I_prec)
            int_sum[i] = int_sum[i] - num_carry * prec
            int_sum[i - 1] = int_sum[i - 1] + num_carry

    positive = True

    for i in range(len(int_sum)):
        if abs(int_sum[i]) > 0:
            if int_sum[i] < 0:
                positive = False
                break

    if positive:
        for i in range(len(int_sum) - 1, 0, -1):
            if int_sum[i] < 0:
                int_sum[i] = int_sum[i] + prec
                int_sum[i - 1] = int_sum[i - 1] - 1

    else:
        for i in range(len(int_sum) - 1, 0, -1):
            if int_sum[i] > 0:
                int_sum[i] = int_sum[i] - prec
                int_sum[i - 1] = int_sum[i - 1] + 1


def _ints_to_real(ints: np.ndarray, pr: list[float]) -> float:
    r = 0.0

    for i in range(len(ints)):
        r = r + pr[i] * ints[i]

    return r


class MPPGlobalSum:
    def __init__(
        self, stencil_factory: StencilFactory, communicator: Communicator
    ) -> None:
        NUMINT = 6
        self._comm = communicator
        self._ints_sum = Quantity(
            data=np.zeros((NUMINT), dtype=Float),
            dims=["K"],
            units="dunno",
            backend=stencil_factory.backend,
        )

        self._ints_sum_reduce = Quantity(
            data=np.zeros((NUMINT), dtype=Float),
            dims=["K"],
            units="dunno",
            backend=stencil_factory.backend,
        )

    def __call__(self, qty_to_sum: Quantity) -> Float:
        NUMBIT = 46
        r_prec = 2.0**NUMBIT
        prec = 2**NUMBIT
        I_prec = 1.0 / (2.0**NUMBIT)
        pr = [
            r_prec**2,
            r_prec,
            1.0,
            1.0 / r_prec,
            1.0 / r_prec**2,
            1.0 / r_prec**3,
        ]
        I_pr = [1.0 / r_prec**2, 1.0 / r_prec, 1.0, r_prec, r_prec**2, r_prec**3]
        prec_error = (2**62 + (2**62 - 1)) / 6
        mag_max_term = 0.0

        # Note: This loop range in i and j are for the TBC test case.
        self._ints_sum[:] = 0
        for j in range(qty_to_sum.field.shape[1]):
            for i in range(qty_to_sum.field.shape[0]):
                _increment_ints_faster(
                    self._ints_sum.data[:],
                    pr,
                    I_pr,
                    qty_to_sum.field[i, j],
                    mag_max_term,
                )

        if not _carry_overflow(self._ints_sum.data, prec, I_prec, prec_error):
            warnings.warn("Overflow in MPP sum", category=UserWarning, stacklevel=2)

        self._comm.all_reduce(
            self._ints_sum,
            ReductionOperator.SUM,
            self._ints_sum_reduce,
        )

        _regularize_ints(self._ints_sum_reduce.data, prec, I_prec)

        return _ints_to_real(self._ints_sum_reduce.data, pr)
