import tempfile
from pathlib import Path

from ndsl.utils import load_f90nml
from pyfv3._config import DynamicalCoreConfig


def tmp_nested_nml() -> list[Path, Path]:
    """Creates a temporary namelist file with an override namelist file nested within.
    Returns main_nml_path, override_nml_path
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir)

        override_nml_contents = """
        &fv_core_nml
            a_imp = 2.0
        /
        """
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".nml"
        ) as override_file:
            override_file.write(override_nml_contents)
            override_file_path = Path(override_file.name)

    main_nml_contents = f"""
        &fv_core_nml
            a_imp = 1.0
            adjust_dry_mass = .false.
            beta = 0.0
            consv_am = .false.
            consv_te = 0.0
            d2_bg = 0.0
            d2_bg_k1 = 0.2
            d2_bg_k2 = 0.1
            d4_bg = 0.15
            d_con = 1.0
            d_ext = 0.0
            dddmp = 0.5
            namelist_override = '{override_file_path}'
    /
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".nml"
    ) as main_file:
        main_file.write(main_nml_contents)
        main_file_path = Path(main_file.name)

    return [main_file_path, override_file_path]


def test_config_from_f90nml_with_override():
    try:
        main_file_path, override_file_path = tmp_nested_nml()
        # Load first the main nml
        main_nml = load_f90nml(main_file_path)
        dycore_config = DynamicalCoreConfig.from_f90nml(main_nml)
        # Spot check that the dycore config values match expected values
        assert dycore_config.a_imp == 2.0  # from override file
        assert dycore_config.d2_bg_k1 == 0.0  # default from DynamicalCoreConfig
        assert dycore_config.c2l_ord == 4  # default from DynamicalCoreConfig
    finally:
        main_file_path.unlink()
        override_file_path.unlink()
