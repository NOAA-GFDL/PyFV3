from typing import TypeAlias
from ndsl import QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.quantity.field_bundle import FieldBundle, FieldBundleType
from pyFV3.version import IS_GEOS

# Defauult maopping for common models
_default_mapping_GEOS = {
    "vapor": 0,
    "liquid": 1,
    "ice": 2,
    "rain": 3,
    "snow": 4,
    "graupel": 5,
    "cloud": 6,
}
_default_mapping_PACE = {
    "vapor": 0,
    "liquid": 1,
    "rain": 2,
    "ice": 3,
    "snow": 4,
    "graupel": 5,
    "om3r": 6,
    "cloud": 7,
}


TracersType: TypeAlias = FieldBundleType.T("Tracers")  # type: ignore


def setup_tracers(
    number_of_tracers: int,
    quantity_factory: QuantityFactory,
    mappings: dict[str, int] | None = None,
) -> FieldBundle:
    """Setup a FieldBundle for tracers. Should be called only once."""

    FieldBundleType.register("Tracers", (number_of_tracers,))

    _unit = "g/kg"
    _dims = [X_DIM, Y_DIM, Z_DIM, "tracers"]

    tracers_qty_factory = FieldBundle.extend_3D_quantity_factory(
        quantity_factory, {"tracers": number_of_tracers}
    )
    data = tracers_qty_factory.zeros(_dims, units=_unit)

    # Some default mappings for ease of use with commonly
    # run models
    if mappings is None:
        if IS_GEOS:
            mappings = _default_mapping_GEOS
        else:
            mappings = _default_mapping_PACE

    return FieldBundle(
        "Tracers",
        quantity=data,
        mapping=mappings,
    )
