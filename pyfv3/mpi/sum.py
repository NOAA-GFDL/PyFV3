import numpy as np

from ndsl import Quantity, QuantityFactory
from ndsl.comm.communicator import Communicator, ReductionOperator
from ndsl.constants import I_DIM, J_DIM
from ndsl.dsl.dace.orchestration import dace_inhibitor
from ndsl.dsl.stencil import GridIndexing
from ndsl.dsl.typing import Float
from ndsl.optional_imports import cupy as cp


class GlobalSum:
    def __init__(
        self,
        quantity_factory: QuantityFactory,
        communicator: Communicator,
        grid_indexing: GridIndexing = None,
    ) -> None:
        self._comm = communicator
        # self._tmp_reduce = quantity_factory.empty(dims=[I_DIM, J_DIM], units="n/a")
        self._tmp_reduce = quantity_factory.zeros(dims=[I_DIM, J_DIM], units="n/a")
        self._isc = grid_indexing.isc
        self._iec = grid_indexing.iec
        self._jsc = grid_indexing.jsc
        self._jec = grid_indexing.jec

    @dace_inhibitor
    def __call__(self, qty_to_sum: Quantity) -> Float:
        assert len(qty_to_sum.field.shape) == 2  # Code handle only 2D quantity
        self._comm.all_reduce(qty_to_sum, ReductionOperator.SUM, self._tmp_reduce)
        if isinstance(self._tmp_reduce[:], np.ndarray):
            return np.sum(
                self._tmp_reduce[self._isc : self._iec + 1, self._jsc : self._jec + 1]
            )
        elif isinstance(self._tmp_reduce[:], cp.ndarray) and cp is not None:
            return cp.sum(
                self._tmp_reduce[self._isc : self._iec + 1, self._jsc : self._jec + 1]
            )
        else:
            raise TypeError("Unsupported array type for reduction result.")
