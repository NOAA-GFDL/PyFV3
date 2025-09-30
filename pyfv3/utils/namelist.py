from ndsl.utils import f90nml_as_dict
from f90nml import Namelist
from pyfv3._config import DynamicalCoreConfig
from dacite import Config, from_dict

DEFAULT_NML_GROUPS = (
    "main_nml",
    "coupler_nml",
    "fv_core_nml",
)

def dycore_config_from_f90nml(nml: Namelist) -> DynamicalCoreConfig:
    nml_dict = f90nml_as_dict(nml, flatten=True, target_groups=DEFAULT_NML_GROUPS)
    dacite_config = Config(type_hooks={tuple[int, int]: tuple[int, int]})
    return from_dict(data_class=DynamicalCoreConfig, data=nml_dict, config=dacite_config)
