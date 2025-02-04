from __future__ import annotations

from typing import Dict, List

import numpy as np

from ndsl import Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM


# FOR REFERENCE - previous descriptive of the tracers, lining up with Pace work
# tracer_variables = [
#     "qvapor",
#     "qliquid",
#     "qrain",
#     "qice",
#     "qsnow",
#     "qgraupel",
#     "qo3mr",
#     "qsgs_tke",
#     "qcld",
# ]


class Tracers:
    unit = "g/kg"
    dims = [X_DIM, Y_DIM, Z_DIM]

    def __init__(self, factory: QuantityFactory) -> None:
        self._quantities: Dict[str, Quantity] = {}
        self._quantity_factory = factory

    def copy_tracer_data(
        self,
        name: str,
        data: np.ndarray,
        unit="unknown",
    ):
        qty = self._quantity_factory.empty(dims=self.dims, units=unit)
        if data.shape > qty.data.shape:
            raise ValueError(
                f"[pyFV3] Tracer {name} size ({data.shape}"
                f" is bigger than grid {qty.data.shape})"
            )
        qty.data[: data.shape[0], : data.shape[1], : data.shape[2]] = data
        self._quantities[name] = qty

    @property
    def count(self) -> int:
        return len(self._quantities)

    def values(self):
        return self._quantities.values()

    def names(self):
        return self._quantities.keys()

    def items(self):
        return self._quantities.items()

    def as_4D_array(self) -> np.ndarray:
        shape = self._quantity_factory.sizer.get_shape(self.dims)
        var4d = np.empty(
            (
                shape[0] - 1,
                shape[1] - 1,
                shape[2] - 1,
                self.count,
            )
        )
        # Skip the extra data point that is meant to align interface
        # and non interface fields
        for idx, q in enumerate(self.values()):
            var4d[:, :, :, idx] = q.data[:-1, :-1, :-1]
        return var4d

    def __getitem__(self, key):
        return self._quantities[key]

    def __setitem__(self, key, value):
        self._quantities[key] = value

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        msg = f"[pyFV3] {self.count} Tracers:\n"
        for q in self.names():
            msg += f"  {q}\n"
        return msg

    @classmethod
    def make(
        cls,
        quantity_factory: QuantityFactory,
        tracer_mapping: List[str],
    ):
        tracers = cls(quantity_factory)
        for name in tracer_mapping:
            qty = quantity_factory.empty(dims=cls.dims, units=cls.unit)
            tracers._quantities[name] = qty
        return tracers

    @staticmethod
    def blind_mapping_from_data(tracer_data: np.ndarray):
        if len(tracer_data.shape) != 4:
            raise ValueError("Expected 4D field as input")
        return [f"Tracer_{idx}" for idx in range(tracer_data.shape[3])]

    @classmethod
    def make_from_4D_array(
        cls,
        quantity_factory: QuantityFactory,
        tracer_mapping: List[str],
        tracer_data: np.ndarray,
    ) -> Tracers:
        if len(tracer_data.shape) != 4:
            raise ValueError("Expected 4D field as input")
        count = len(tracer_mapping)
        if count > tracer_data.shape[3]:
            raise ValueError(
                f"Mapping size {len(tracer_mapping)} is bigger than"
                f" data dimensionality {tracer_data.shape[3]}"
            )
        tracers = cls(quantity_factory)
        for idx in range(0, count):
            tracers.copy_tracer_data(
                name=tracer_mapping[idx] or f"Tracer_{idx}",
                data=tracer_data[:, :, :, idx],
                unit=cls.unit,
            )
        return tracers
