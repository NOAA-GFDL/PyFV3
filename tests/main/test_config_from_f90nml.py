from datetime import timedelta
from math import floor
from typing import List

import f90nml
import pytest

from pyfv3 import DynamicalCoreConfig


TESTED_CONFIGS: List[str] = [
    "examples/configs/c48_baroclinic_stable_input.nml",
]


@pytest.mark.parametrize(
    "tested_configs",
    [
        pytest.param(TESTED_CONFIGS, id="example f90nml configs"),
    ],
)
def test_config_from_f90nml(tested_configs: List[str]):
    """TODO description"""
    for config_file in tested_configs:
        config = f90nml.read(config_file)
        runtime = {
            "days": 0.0,
            "hours": 0.0,
            "minutes": 0.0,
            "seconds": 0.0,
        }

        for key in runtime.keys():
            if key in config["main_nml"].keys():
                runtime[key] = config["main_nml"][key]
        total_time = timedelta(
            days=runtime["days"],
            hours=runtime["hours"],
            minutes=runtime["minutes"],
            seconds=runtime["seconds"],
        )
        timestep = timedelta(seconds=config["main_nml"]["dt_atmos"])
        n_steps = floor(total_time.total_seconds() / timestep.total_seconds())
        dycore_config = DynamicalCoreConfig.from_f90nml(config)
        default_dycore_config = DynamicalCoreConfig()

        assert dycore_config.dt_atmos == config["main_nml"]["dt_atmos"]
        assert dycore_config.n_steps == n_steps
        assert dycore_config.adiabatic == default_dycore_config.adiabatic
