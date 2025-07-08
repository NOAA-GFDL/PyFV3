import os
from datetime import timedelta
from math import floor
from typing import List

import pytest
import yaml

from pyfv3 import DynamicalCoreConfig


TESTED_CONFIGS: List[str] = [
    "/pyFV3/examples/configs/c12_baroclinic.yaml",
]


@pytest.mark.parametrize(
    "tested_configs",
    [
        pytest.param(TESTED_CONFIGS, id="example configs"),
    ],
)
def test_config_from_yaml(tested_configs: List[str]):
    for config_file in tested_configs:
        with open(os.path.abspath(config_file), "r") as f:
            config = yaml.safe_load(f)
        runtime = {
            "days": 0.0,
            "hours": 0.0,
            "minutes": 0.0,
            "seconds": 0.0,
        }
        for key in runtime.keys():
            if key in config.keys():
                runtime[key] = config[key]
        total_time = timedelta(
            days=runtime["days"],
            hours=runtime["hours"],
            minutes=runtime["minutes"],
            seconds=runtime["seconds"],
        )
        timestep = timedelta(seconds=config["dt_atmos"])
        n_steps = floor(total_time.total_seconds() / timestep.total_seconds())
        dycore_config = DynamicalCoreConfig.from_yaml(config_file)
        assert dycore_config.n_steps == n_steps
