from ndsl import QuantityFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.quantity.field_bundle import FieldBundle, FieldBundleType
from pyfv3.version import IS_GEOS


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


TracersType = FieldBundleType.T("Tracers")

_mappings: dict[str, int] = {}


def setup_tracers(
    number_of_tracers: int,
    quantity_factory: QuantityFactory,
    mappings: dict[str, int] | None = None,
) -> None:
    global _mappings

    FieldBundleType.register("Tracers", (number_of_tracers,))

    # Some default mappings for ease of use with commonly
    # run models
    if mappings is None:
        if IS_GEOS:
            mappings = _default_mapping_GEOS
        else:
            mappings = _default_mapping_PACE

    _mappings = mappings
    quantity_factory.add_data_dimensions({"tracers": number_of_tracers})


def make_tracers(quantity_factory: QuantityFactory) -> FieldBundle:
    """Setup a FieldBundle for tracers. Should be called only once."""
    global _mappings  # noqa

    _unit = "g/kg"
    _dims = [I_DIM, J_DIM, K_DIM, "tracers"]

    data = quantity_factory.zeros(_dims, units=_unit)

    return FieldBundle("Tracers", quantity=data, mapping=_mappings)
