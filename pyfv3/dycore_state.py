from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Self

import numpy.typing as npt
import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl import Backend, GridSizer, Quantity, QuantityFactory
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    K_DIM,
    K_INTERFACE_DIM,
)
from ndsl.dsl.typing import Float
from ndsl.quantity.field_bundle import FieldBundle
from ndsl.restart._legacy_restart import open_restart
from ndsl.typing import Communicator
from pyfv3.tracers import TracersType, setup_tracers


DEFAULT_TRACER_PROPERTIES = {
    "specific_humidity": {
        "pyFV3_key": "vapor",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "sphum",
        "units": "g/kg",
    },
    "cloud_liquid_water_mixing_ratio": {
        "pyFV3_key": "liquid",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "liq_wat",
        "units": "g/kg",
    },
    "cloud_ice_mixing_ratio": {
        "pyFV3_key": "ice",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "ice_wat",
        "units": "g/kg",
    },
    "rain_mixing_ratio": {
        "pyFV3_key": "rain",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "rainwat",
        "units": "g/kg",
    },
    "snow_mixing_ratio": {
        "pyFV3_key": "snow",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "snowwat",
        "units": "g/kg",
    },
    "graupel_mixing_ratio": {
        "pyFV3_key": "graupel",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "graupel",
        "units": "g/kg",
    },
    "ozone_mixing_ratio": {
        "pyFV3_key": "o3mr",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "o3mr",
        "units": "g/kg",
    },
    "turbulent_kinetic_energy": {
        "pyFV3_key": "sgs_tke",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "sgs_tke",
        "units": "g/kg",
    },
    "cloud_fraction": {
        "pyFV3_key": "cloud",
        "dims": [K_DIM, J_DIM, I_DIM],
        "restart_name": "cld_amt",
        "units": "g/kg",
    },
}


@dataclass()
class DycoreState:
    u: Quantity = field(
        metadata={
            "name": "x_wind",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    v: Quantity = field(
        metadata={
            "name": "y_wind",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    w: Quantity = field(
        metadata={
            "name": "vertical_wind",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    # TODO: move a-grid winds to temporary internal storage
    ua: Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
        }
    )
    uc: Quantity = field(
        metadata={
            "name": "x_wind_on_c_grid",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    vc: Quantity = field(
        metadata={
            "name": "y_wind_on_c_grid",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    delp: Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: Quantity = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    ps: Quantity = field(
        metadata={
            "name": "surface_pressure",
            "dims": [I_DIM, J_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    pe: Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "Pa",
            "n_halo": 1,
            "intent": "inout",
        }
    )
    pt: Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    peln: Quantity = field(
        metadata={
            "name": "logarithm_of_interface_pressure",
            "dims": [
                I_DIM,
                J_DIM,
                K_INTERFACE_DIM,
            ],
            "units": "ln(Pa)",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pk: Quantity = field(
        metadata={
            "name": "interface_pressure_raised_to_power_of_kappa",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    pkz: Quantity = field(
        metadata={
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    tracers: TracersType = field(
        metadata={
            "name": "tracers",
            "units": "g/kg",
            "intent": "inout",
        }
    )
    q_con: Quantity = field(
        metadata={
            "name": "total_condensate_mixing_ratio",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    omga: Quantity = field(
        metadata={
            "name": "vertical_pressure_velocity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa/s",
            "intent": "inout",
        }
    )
    mfxd: Quantity = field(
        metadata={
            "name": "accumulated_x_mass_flux",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    mfyd: Quantity = field(
        metadata={
            "name": "accumulated_y_mass_flux",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "unknown",
            "n_halo": 0,
            "intent": "inout",
        }
    )
    cxd: Quantity = field(
        metadata={
            "name": "accumulated_x_courant_number",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "",
            "n_halo": (0, 3),
            "intent": "inout",
        }
    )
    cyd: Quantity = field(
        metadata={
            "name": "accumulated_y_courant_number",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "",
            "n_halo": (3, 0),
            "intent": "inout",
        }
    )
    diss_estd: Quantity = field(
        metadata={
            "name": "dissipation_estimate_from_heat_source",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "unknown",
            "n_halo": (3, 3),
            "intent": "inout",
        }
    )
    """
    how much energy is dissipated, is mainly captured
    to send to the stochastic physics (in contrast to heat_source)
    """
    phis: Quantity = field(
        metadata={
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [I_DIM, J_DIM],
            "intent": "in",
        }
    )
    bdt: float = field(default=0.0)
    mdt: float = field(default=0.0)

    def __post_init__(self) -> None:
        for _field in fields(self):
            if _field.name == "tracers":
                continue
            for check_name in ["units", "dims"]:
                if check_name in _field.metadata:
                    required = _field.metadata[check_name]
                    actual = getattr(getattr(self, _field.name), check_name)
                    if isinstance(required, list):
                        actual = list(actual)
                    if actual != required:
                        raise TypeError(
                            f"{_field.name} has metadata {check_name} of {actual}"
                            f"that does not match the requirement {required}"
                        )

    @classmethod
    def init_zeros(
        cls,
        quantity_factory: QuantityFactory,
        tracer_count: int,
        backend: Backend,
        dtype_dict: dict[str, type] | None = None,
        allow_mismatch_float_precision: bool = False,
    ) -> Self:
        initial_storages = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_storages[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"],
                    _field.metadata["units"],
                    dtype=(
                        dtype_dict[_field.name]
                        if dtype_dict and _field.name in dtype_dict.keys()
                        else Float
                    ),
                    allow_mismatch_float_precision=allow_mismatch_float_precision,
                ).data
            elif _field.name == "tracers":
                qty_factory_tracers = FieldBundle.extend_3D_quantity_factory(
                    quantity_factory, {"tracers": tracer_count}
                )
                initial_storages[_field.name] = qty_factory_tracers.zeros(
                    [I_DIM, J_DIM, K_DIM, "tracers"],
                    _field.metadata["units"],
                    dtype=Float,
                ).data
        return cls.init_from_storages(
            storages=initial_storages,
            quantity_factory=quantity_factory,
            backend=backend,
        )

    @classmethod
    def init_from_numpy_arrays(
        cls,
        dict_of_numpy_arrays: dict,
        sizer: GridSizer,
        backend: Backend,
        tracer_list: list[str],
        quantity_factory: QuantityFactory,
    ) -> Self:
        field_names = [_field.name for _field in fields(cls)]
        for variable_name in dict_of_numpy_arrays.keys():
            if variable_name not in field_names:
                raise KeyError(
                    variable_name + " is provided, but not part of the dycore state"
                )
        dict_state = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                dict_state[_field.name] = Quantity(
                    dict_of_numpy_arrays[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=quantity_factory.sizer.get_origin(dims),
                    extent=quantity_factory.sizer.get_extent(dims),
                    backend=backend,
                )
            elif issubclass(_field.type, TracersType):
                if len(dict_of_numpy_arrays[_field.name]) != len(tracer_list):
                    raise ValueError(
                        "[pyFV3] DycoreState init:"
                        f" tracer list size ({len(tracer_list)})"
                        " doesn't match the inputs size"
                        f" ({len(dict_of_numpy_arrays[_field.name])})"
                    )
                dict_state[_field.name] = Tracers.make(
                    quantity_factory=quantity_factory,
                    tracer_mapping=tracer_list,
                )
        state = cls(**dict_state)
        return state

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        quantity_factory: QuantityFactory,
        bdt: float = 0.0,
        mdt: float = 0.0,
        backend: Backend | None = None,
    ) -> Self:
        if not backend:
            backend = Backend.python()
        inputs = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                quantity = Quantity(
                    storages[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=quantity_factory.sizer.get_origin(dims),
                    extent=quantity_factory.sizer.get_extent(dims),
                    backend=backend,
                )
                inputs[_field.name] = quantity
            elif "tracers" == _field.name:
                tracers = setup_tracers(storages["tracers"].shape[3], quantity_factory)
                tracers.quantity.data[:] = storages["tracers"][:]
                inputs[_field.name] = tracers

        return cls(**inputs, bdt=bdt, mdt=mdt)

    @classmethod
    def from_fortran_restart(
        cls,
        *,
        quantity_factory: QuantityFactory,
        communicator: Communicator,
        path: str,
        backend: Backend,
    ) -> Self:
        state_dict: Mapping[str, Quantity] = open_restart(
            dirname=path,
            communicator=communicator,
            tracer_properties=DEFAULT_TRACER_PROPERTIES,
        )
        new = cls.init_zeros(
            quantity_factory=quantity_factory,
            tracer_count=len(DEFAULT_TRACER_PROPERTIES),
            backend=backend,
        )
        new.pt.view[:] = new.pt.np.asarray(
            state_dict["air_temperature"].transpose(new.pt.dims).view[:]
        )
        new.delp.view[:] = new.delp.np.asarray(
            state_dict["pressure_thickness_of_atmospheric_layer"]
            .transpose(new.delp.dims)
            .view[:]
        )
        new.phis.view[:] = new.phis.np.asarray(
            state_dict["surface_geopotential"].transpose(new.phis.dims).view[:]
        )
        new.w.view[:] = new.w.np.asarray(
            state_dict["vertical_wind"].transpose(new.w.dims).view[:]
        )
        new.u.view[:] = new.u.np.asarray(
            state_dict["x_wind"].transpose(new.u.dims).view[:]
        )
        new.v.view[:] = new.v.np.asarray(
            state_dict["y_wind"].transpose(new.v.dims).view[:]
        )
        new.tracers.vapor.view[:] = new.tracers.vapor.np.asarray(
            state_dict["specific_humidity"].transpose(new.tracers.vapor.dims).view[:]
        )
        new.tracers.liquid.view[:] = new.tracers.liquid.np.asarray(
            state_dict["cloud_liquid_water_mixing_ratio"]
            .transpose(new.tracers.liquid.dims)
            .view[:]
        )
        new.tracers.ice.view[:] = new.tracers.ice.np.asarray(
            state_dict["cloud_ice_mixing_ratio"].transpose(new.tracers.ice.dims).view[:]
        )
        new.tracers.rain.view[:] = new.tracers.rain.np.asarray(
            state_dict["rain_mixing_ratio"].transpose(new.tracers.rain.dims).view[:]
        )
        new.tracers.snow.view[:] = new.tracers.snow.np.asarray(
            state_dict["snow_mixing_ratio"].transpose(new.tracers.snow.dims).view[:]
        )
        new.tracers.graupel.view[:] = new.tracers.graupel.np.asarray(
            state_dict["graupel_mixing_ratio"]
            .transpose(new.tracers.graupel.dims)
            .view[:]
        )
        new.tracers.o3mr.view[:] = new.tracers.o3mr.np.asarray(
            state_dict["ozone_mixing_ratio"].transpose(new.tracers.o3mr.dims).view[:]
        )
        new.tracers.cloud.view[:] = new.tracers.cld.np.asarray(
            state_dict["cloud_fraction"].transpose(new.tracers.cld.dims).view[:]
        )
        new.delz.view[:] = new.delz.np.asarray(
            state_dict["vertical_thickness_of_atmospheric_layer"]
            .transpose(new.delz.dims)
            .view[:]
        )

        return new

    def _xr_dataarray_from_array(
        self, name: str, metadata: MappingProxyType[Any, Any], data: npt.ArrayLike
    ):
        dims = [f"{dim_name}_{name}" for dim_name in metadata["dims"]]
        return xr.DataArray(
            gt_utils.asarray(data),
            dims=dims,
            attrs={
                "long_name": metadata["name"],
                "units": metadata.get("units", "unknown"),
            },
        )

    @property
    def xr_dataset(self) -> xr.Dataset:
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if issubclass(field_info.type, Quantity):
                data_vars[name] = self._xr_dataarray_from_array(
                    name=name,
                    metadata=field_info.metadata,
                    data=getattr(self, name).data,
                )
            if isinstance(field_info.type, FieldBundle):
                data_vars[name] = self._xr_dataarray_from_array(
                    name=name,
                    metadata=field_info.metadata,
                    data=getattr(self, name).quantity.data,
                )
        return xr.Dataset(data_vars=data_vars)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def as_dict(self, quantity_only: bool = True) -> dict[str, Quantity | int]:
        if quantity_only:
            return {k: v for k, v in asdict(self).items() if isinstance(v, Quantity)}
        else:
            return {k: v for k, v in asdict(self).items()}
