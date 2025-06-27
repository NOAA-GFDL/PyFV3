from ndsl import Quantity, QuantityFactory
from ndsl.dsl.typing import Float
from ndsl.comm.communicator import Communicator, ReductionOperator
from ndsl.constants import X_DIM, Y_DIM
from ndsl.dsl.dace.orchestration import dace_inhibitor


class GlobalSum:
    def __init__(
        self, quantity_factory: QuantityFactory, communicator: Communicator
    ) -> None:
        self._comm = communicator
        self._tmp_reduce = quantity_factory.empty(dims=[X_DIM, Y_DIM], units="n/a")

    @dace_inhibitor
    def __call__(self, qty_to_sum: Quantity) -> Float:
        assert len(qty_to_sum.field.shape) == 2  # Code handle only 2D quantity
        self._comm.all_reduce(qty_to_sum, ReductionOperator.SUM, self._tmp_reduce)
        return qty_to_sum.field.sum(axis=0).sum(axis=0)
