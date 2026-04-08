from ndsl import QuantityFactory
from ndsl.dsl.gt4py_utils import run_once
from ndsl.dsl.typing import Float
from ndsl.quantity.data_dimensions_field import DataDimensionsField, SparseNameMapping


FVTracers = DataDimensionsField.declare()
FVTracersAxisName = "fv_tracers"

_EXPECTED_FV_TRACERS = [
    "vapor",
    "liquid",
    "rain",
    "ice",
    "snow",
    "graupel",
    "cloud",
    "o3mr",
    "sgs_tke",
]
"""Expected tracers for FV dynamics to be able to run in the current state."""


def setup_fvtracers(
    quantity_factory: QuantityFactory,
    tracer_count: int,
    name_mapping: SparseNameMapping,
) -> None:
    """Setup FV Tracers and sparse mapping to call tracer by name"""

    if not all(tracer in name_mapping for tracer in _EXPECTED_FV_TRACERS):
        raise ValueError(
            f"FV Tracers requires name mapping for all of the follwoing {_EXPECTED_FV_TRACERS}."
            f"Given {name_mapping}."
        )

    quantity_factory.add_data_dimensions({FVTracersAxisName: tracer_count})
    DataDimensionsField.register(
        FVTracers, quantity_factory, [FVTracersAxisName], name_mapping, dtype=Float
    )


@run_once
def default_ai2_tracers(quantity_factory: QuantityFactory) -> None:
    """Default FV Tracers setup for the AI2 dataset & code"""
    ai2_tracers = {
        "vapor": 0,
        "liquid": 1,
        "rain": 2,
        "ice": 3,
        "snow": 4,
        "graupel": 5,
        "o3mr": 6,
        "sgs_tke": 7,
        "cloud": 8,
    }
    setup_fvtracers(quantity_factory, len(ai2_tracers.keys()), ai2_tracers)
