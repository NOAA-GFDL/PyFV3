from dacite import Config, from_dict
from f90nml import Namelist

from ndsl.utils import f90nml_as_dict, load_f90nml_as_dict
from pyfv3._config import DynamicalCoreConfig


DEFAULT_NML_GROUPS = (
    "main_nml",
    "coupler_nml",
    "fv_core_nml",
)


def dycore_config_from_f90nml(
    nml: Namelist, use_default_groups: bool = True
) -> DynamicalCoreConfig:
    """Uses the nml to create a DynamicalCoreConfig.
        Only the DEFAULT_NML_GROUPS from the nml are considered
        when initializing the DynamicalCoreConfig. If the nml
        has a 'namelist_override' key, then that will be used to
        load an additional namelist file to override the
        DynamicalCoreConfig values.

    Args:
        nml: f90nml.Namelist
        use_default_groups: if True, the DEFAULT_NML_GROUPS will
                            be used for initializing the config.
                            Otherwise, parameters from all groups
                            will be used to initialize.
    """
    if use_default_groups:
        target_groups = DEFAULT_NML_GROUPS
    else:
        target_groups = None
    nml_dict = f90nml_as_dict(nml, flatten=True, target_groups=target_groups)
    dacite_config = Config(type_hooks={tuple[int, int]: tuple[int, int]})
    dycore_config = from_dict(
        data_class=DynamicalCoreConfig, data=nml_dict, config=dacite_config
    )

    # Patch if a namelist_override exists
    # NOTE: We're doing the patching of the dycore based on the override_nml
    # here, rather than DynamicalCoreConfig.__post_init__ as one way to avoid
    # circular dependencies.
    if dycore_config.namelist_override is not None:
        override_dict = load_f90nml_as_dict(
            dycore_config.namelist_override, flatten=True, target_groups=target_groups
        )
        for key, value in override_dict.items():
            if key in dycore_config.__dataclass_fields__:
                setattr(dycore_config, key, value)
    return dycore_config
