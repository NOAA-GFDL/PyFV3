from enum import Enum

import pyFV3.initialization.test_cases.initialize_baroclinic as bc
import pyFV3.initialization.test_cases.initialize_rossby as rossby
import pyFV3.initialization.test_cases.initialize_tc as tc
from ndsl import CubedSphereCommunicator, MetaEnumStr, QuantityFactory
from ndsl.grid import GridData
from ndsl.typing import Communicator
from pyFV3.dycore_state import DycoreState


class Cases(Enum, metaclass=MetaEnumStr):
    baroclinic = "baroclinic"
    rossby = "rossby"
    tropicalcyclone = "tropicalcyclone"


def init_analytic_state(
    analytic_init_case: str,
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: Communicator,
) -> DycoreState:
    """
    This method initializes the chosen analytic test case type
    Args:
        analytic_init_str:      test case specifier
        grid_data:              current selected grid data values
        quantity_factory:       inclusion of QuantityFactory class
        adiabatic:              flag for adiabatic methods
        hydrostatic:            flag for hydrostatic methods
        moist_phys:             flag for including moisture physics methods
        comm:                   inclusion of CubedSphereCommunicator class

    Returns:
        an instance of DycoreState class
    """
    # Cases that expect Cubed Sphere Communicator
    spherical_cases = [
        Cases.baroclinic.value,
        Cases.tropicalcyclone.value,
        Cases.rossby.value,
    ]

    if analytic_init_case in spherical_cases:  # type: ignore
        # TODO: Consider CubedSphereCommunicator check within individual init_*() calls
        if not isinstance(comm, CubedSphereCommunicator):
            raise TypeError(
                f"Expected CubedSphereCommunicator instance for 'comm', "
                f"got {type(comm).__name__} instead."
            )

        if analytic_init_case == Cases.baroclinic.value:  # type: ignore
            return bc.init_baroclinic_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                adiabatic=adiabatic,
                hydrostatic=hydrostatic,
                moist_phys=moist_phys,
                comm=comm,
            )
        elif analytic_init_case == Cases.tropicalcyclone.value:  # type: ignore
            return tc.init_tc_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                hydrostatic=hydrostatic,
                comm=comm,
            )
        elif analytic_init_case == Cases.rossby.value:  # type: ignore
            return rossby.init_rossby_state(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                comm=comm,
            )
        else:
            raise ValueError(f"Case {analytic_init_case} not implemented")
    else:
        raise ValueError(f"Case {analytic_init_case} not recognized")
