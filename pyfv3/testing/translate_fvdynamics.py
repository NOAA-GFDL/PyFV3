from dataclasses import fields
from datetime import timedelta
from typing import Any

import numpy as np
import pytest
from f90nml import Namelist

import ndsl.dsl.gt4py_utils as utils
from ndsl import FieldBundle, Quantity, QuantityFactory, StencilFactory
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    K_DIM,
    K_INTERFACE_DIM,
)
from ndsl.grid import GridData
from ndsl.performance import NullTimer
from ndsl.stencils.testing import Grid, ParallelTranslateBaseSlicing
from ndsl.typing import Communicator
from pyfv3._config import DynamicalCoreConfig
from pyfv3.dycore_state import DycoreState
from pyfv3.stencils import fv_dynamics


class TranslateFVDynamics(ParallelTranslateBaseSlicing):
    compute_grid_option = True
    inputs: dict[str, Any] = {
        "q_con": {
            "name": "total_condensate_mixing_ratio",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "kg/kg",
        },
        "delp": {
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa",
        },
        "delz": {
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m",
        },
        "ps": {
            "name": "surface_pressure",
            "dims": [I_DIM, J_DIM],
            "units": "Pa",
        },
        "pe": {
            "name": "interface_pressure",
            "dims": [I_DIM, K_INTERFACE_DIM, J_DIM],
            "units": "Pa",
            "n_halo": 1,
        },
        "ak": {
            "name": "atmosphere_hybrid_a_coordinate",
            "dims": [K_INTERFACE_DIM],
            "units": "Pa",
        },
        "bk": {
            "name": "atmosphere_hybrid_b_coordinate",
            "dims": [K_INTERFACE_DIM],
            "units": "",
        },
        "pk": {
            "name": "interface_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "n_halo": 0,
        },
        "pkz": {
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [I_DIM, J_DIM, K_DIM],
            "n_halo": 0,
        },
        "peln": {
            "name": "logarithm_of_interface_pressure",
            "units": "ln(Pa)",
            "dims": [I_DIM, K_INTERFACE_DIM, J_DIM],
            "n_halo": 0,
        },
        "mfxd_FV": {
            "name": "accumulated_x_mass_flux",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "mfyd_FV": {
            "name": "accumulated_y_mass_flux",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "cxd_FV": {
            "name": "accumulated_x_courant_number",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "",
            "n_halo": (0, 3),
        },
        "cyd_FV": {
            "name": "accumulated_y_courant_number",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "",
            "n_halo": (3, 0),
        },
        "diss_estd": {
            "name": "dissipation_estimate_from_heat_source",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "unknown",
        },
        "pt": {
            "name": "air_temperature",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "degK",
        },
        "u": {
            "name": "x_wind",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "m/s",
        },
        "v": {
            "name": "y_wind",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "ua": {
            "name": "eastward_wind",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "va": {
            "name": "northward_wind",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "uc": {
            "name": "x_wind_on_c_grid",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "vc": {
            "name": "y_wind_on_c_grid",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "m/s",
        },
        "w": {
            "name": "vertical_wind",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "phis": {
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [I_DIM, J_DIM],
        },
        "omga": {
            "name": "vertical_pressure_velocity",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa/s",
        },
        "bdt": {"dims": []},
        "ptop": {"dims": []},
    }

    outputs = inputs.copy()
    outputs["tracers"] = {}

    for name in ("bdt", "ak", "bk", "ptop", "ua"):
        outputs.pop(name)

    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
        *args: Any,
        **kwargs: dict,
    ) -> None:
        super().__init__(grid, namelist, stencil_factory, *args, **kwargs)
        fv_dynamics_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "ps": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pkz": grid.compute_dict(),
            "phis": {},
            "q_con": {},
            "delp": {},
            "pt": {},
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd_FV": grid.x3d_compute_dict(),
            "mfyd_FV": grid.y3d_compute_dict(),
            "cxd_FV": grid.x3d_compute_domain_y_dict(),
            "cyd_FV": grid.y3d_compute_domain_x_dict(),
            "diss_estd": {},
        }
        self._base.in_vars["data_vars"].update(fv_dynamics_vars)

        self._base.out_vars.update(fv_dynamics_vars)
        self._base.out_vars["ps"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}
        self._base.out_vars["phis"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}
        self._base.out_vars["tracers"] = {}
        self._base.out_vars.pop("ua")

        self.max_error = 1e-5

        self.ignore_near_zero_errors: dict[str, float | bool] = {}
        self.dycore: fv_dynamics.DynamicalCore | None = None
        self.stencil_factory = stencil_factory
        self._quantity_factory = QuantityFactory(
            sizer=grid.sizer,
            backend=stencil_factory.backend,
        )
        self.namelist: DynamicalCoreConfig = DynamicalCoreConfig.from_f90nml(namelist)

    def state_from_inputs(self, inputs: dict[str, np.ndarray]) -> DycoreState:
        tracers = self._quantity_factory.empty(
            (
                inputs["tracers"].shape[0] + 1,
                inputs["tracers"].shape[1] + 1,
                inputs["tracers"].shape[2] + 1,
                inputs["tracers"].shape[3],
            )
        )
        tracers[:-1, :-1, :-1, :] = inputs.pop("tracers")
        input_storages = super().state_from_inputs(inputs)
        input_storages["tracers"] = tracers
        # Move fluxes and courant numbers
        input_storages["mfxd"] = input_storages.pop("mfxd_FV")
        input_storages["mfyd"] = input_storages.pop("mfyd_FV")
        input_storages["cxd"] = input_storages.pop("cxd_FV")
        input_storages["cyd"] = input_storages.pop("cyd_FV")
        # making sure we init DycoreState with the exact set of variables
        accepted_keys = [_field.name for _field in fields(DycoreState)]
        to_delete = []
        for name in input_storages.keys():
            if name not in accepted_keys:
                to_delete.append(name)
        for name in to_delete:
            del input_storages[name]
        state = DycoreState.init_from_storages(
            storages=input_storages,
            quantity_factory=self._quantity_factory,
        )
        return state

    def prepare_data(self, inputs: dict) -> tuple[DycoreState, GridData]:
        for name in ("ak", "bk"):
            inputs[name] = utils.make_storage_data(
                inputs[name],
                inputs[name].shape,
                len(inputs[name].shape) * (0,),
                backend=self.stencil_factory.backend,
            )
        grid_data = self.grid.grid_data
        # These aren't in the Grid-Info savepoint, but are in the generated grid
        if grid_data.ak is None or grid_data.bk is None:
            grid_data.ak = inputs["ak"]
            grid_data.bk = inputs["bk"]
            grid_data.ptop = inputs["ptop"]

        state = self.state_from_inputs(inputs)
        return state, grid_data

    def compute_parallel(self, inputs: dict, communicator: Communicator) -> dict:
        state, grid_data = self.prepare_data(inputs)
        self.dycore = fv_dynamics.DynamicalCore(
            comm=communicator,
            grid_data=grid_data,
            stencil_factory=self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            damping_coefficients=self.grid.damping_coefficients,
            config=self.config,
            phis=state.phis,
            state=state,
            exclude_tracers=["cloud"],
            timestep=timedelta(seconds=float(inputs["bdt"])),
        )
        self.dycore.step_dynamics(state, NullTimer())
        outputs = self.outputs_from_state(state)
        return outputs

    def outputs_from_state(self, state: DycoreState) -> dict:
        if len(self.outputs) == 0:
            return {}

        outputs = {}
        storages = {}
        for name, _properties in self.outputs.items():
            if name in ["mfxd_FV", "mfyd_FV", "cxd_FV", "cyd_FV"]:
                storages[name] = state[name[:-3]].data
            elif isinstance(state[name], FieldBundle):
                storages[name] = state[name].quantity.data
            elif isinstance(state[name], Quantity):
                storages[name] = state[name].data
            elif len(self.outputs[name]["dims"]) > 0:
                storages[name] = state[name]  # assume it's a storage
            else:
                outputs[name] = state[name]  # scalar
        outputs.update(self._base.slice_output(storages))
        return outputs

    def compute_sequential(self, *args: Any, **kwargs: dict) -> None:
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, not running in mock-parallel"
        )
