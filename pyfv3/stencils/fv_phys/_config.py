import dataclasses

import ndsl.constants as constants
from ndsl.dsl.typing import Float


DEFAULT_INT = 0
DEFAULT_STR = ""
DEFAULT_BOOL = False
DEFAULT_FLOAT = Float(0.0)

@dataclasses.dataclass
class GrayRadiationConfig:
    prog_low_cloud: bool = True
    """ Whether to calculate low cloud fraction """
    low_cf0: Float = 0.3
    """ global mean *low* cloud fraction """
    dt_atmos: Float = DEFAULT_FLOAT
    """ model timestep in seconds """
    diurnal_cycle: bool = False
    """ Whether to apply diurnal cycle to solar flux """
    sw_abs: Float = DEFAULT_FLOAT
    """ fraction of the solar absorbed/reflected by the atm """
    shift_n: Float = 12.0
    strat_off: bool = True
    """ Cooler stratosphere, from SJ """
    heating_rate: Float = 0.5
    """ deg K per day, stratosphere """
    strat_rad: bool = False
    """ whether to do Solar heating above 100 mb """