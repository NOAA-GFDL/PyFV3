from typing import Any

from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM
from pyfv3.stencils.map_single import MapSingle


class MapSingleFactory:
    _object_pool: dict[tuple[int, int, tuple[str, ...]], MapSingle] = {}
    """Pool of MapSingle objects."""

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ) -> None:
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory

    def __call__(
        self,
        kord: int,
        mode: int,
        *args: Any,
        **kwargs: dict,
    ) -> None:
        key_tuple = (kord, mode, (X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM))
        if key_tuple not in self._object_pool:
            self._object_pool[key_tuple] = MapSingle(
                self.stencil_factory, self.quantity_factory, *key_tuple
            )
        return self._object_pool[key_tuple](*args, **kwargs)
