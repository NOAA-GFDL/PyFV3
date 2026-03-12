from types import SimpleNamespace

from f90nml import Namelist

from ndsl import Quantity, StencilFactory
from ndsl.constants import (
    I_DIM,
    I_INTERFACE_DIM,
    J_DIM,
    J_INTERFACE_DIM,
    K_DIM,
    K_INTERFACE_DIM,
)
from ndsl.dsl.typing import Float
from ndsl.stencils.testing import Grid, ParallelTranslateBaseSlicing
from pyfv3 import DynamicalCoreConfig
from pyfv3.stencils.remapping_GEOS import LagrangianToEulerian_GEOS
from pyfv3.tracers import TracersType, setup_tracers


class TranslateRemapping_GEOS(ParallelTranslateBaseSlicing):
    inputs = {
        "pe": {
            "name": "pe",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "No Units",
        },
        "delp": {
            "name": "delp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "delz": {
            "name": "delz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "q_con": {
            "name": "q_con",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "pt": {
            "name": "pt",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "cappa": {
            "name": "cappa",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "ps": {
            "name": "ps",
            "dims": [I_DIM, J_DIM],
            "units": "No Units",
        },
        "peln": {
            "name": "peln",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "No Units",
        },
        "ak": {
            "name": "ak",
            "dims": [K_INTERFACE_DIM],
            "units": "No Units",
        },
        "bk": {
            "name": "bk",
            "dims": [K_INTERFACE_DIM],
            "units": "No Units",
        },
        "pk": {
            "name": "pk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "pkz": {
            "name": "pkz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "w": {
            "name": "w",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "u": {
            "name": "u",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "No Units",
        },
        "v": {
            "name": "v",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "mfy_R4": {
            "name": "mfy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "cy_R4": {
            "name": "cy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "mfx_R4": {
            "name": "mfx",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "cx_R4": {
            "name": "cx",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "phis": {
            "name": "phis",
            "dims": [I_DIM, J_DIM],
            "units": "No Units",
        },
        "te_2d": {
            "name": "te_2d",
            "dims": [I_DIM, J_DIM],
            "units": "No Units",
        },
        "wsd": {
            "name": "wsd",
            "dims": [I_DIM, J_DIM],
            "units": "No Units",
        },
        "dp1": {
            "name": "dp1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "pfull": {
            "name": "pfull",
            "dims": [K_DIM],
            "units": "No Units",
        },
    }
    outputs = {
        "pt": {
            "name": "pt",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "cappa": {
            "name": "cappa",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "delp": {
            "name": "delp",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "delz": {
            "name": "delz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "w": {
            "name": "w",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "u": {
            "name": "u",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "No Units",
        },
        "v": {
            "name": "v",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "mfy_R4": {
            "name": "mfy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "cy_R4": {
            "name": "cy",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "mfx_R4": {
            "name": "mfx",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "cx_R4": {
            "name": "cx",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "peln": {
            "name": "peln",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "No Units",
        },
        "pe": {
            "name": "pe",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "No Units",
        },
        "pk": {
            "name": "pk",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "pkz": {
            "name": "pkz",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "q_con": {
            "name": "q_con",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "dp1": {
            "name": "dp1",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "No Units",
        },
        "ps": {
            "name": "ps",
            "dims": [I_DIM, J_DIM],
            "units": "No Units",
        },
    }

    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self._base.in_vars["data_vars"] = {
            "tracers": {},
            "w": {
                "kend": grid.npz - 1,
            },
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "delz": {},
            "pt": {},
            "dp1": {},
            "delp": {},
            "cappa": {},
            "q_con": {},
            "pkz": grid.compute_dict(),
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kaxis": 1,
                "kend": grid.npz,
            },
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "ps": {},
            "wsd": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "mfy_R4": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "cy_R4": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "mfx_R4": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "cx_R4": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
            "phis": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "te_2d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            # column variables...
            "ak": {},
            "bk": {},
            "pfull": grid.compute_buffer_k_dict(),
        }
        self._base.in_vars["parameters"] = [
            "ptop",
            "akap",
            "zvir",
            "last_step",
            "consv_te",
            "mdt",
            "nq",
        ]
        self._base.out_vars = {}
        for k in [
            "tracers",
            "pe",
            "pkz",
            "pk",
            "peln",
            "pt",
            "cappa",
            "delp",
            "delz",
            "q_con",
            "u",
            "v",
            "w",
            "ps",
            "dp1",
            "mfy_R4",
            "cy_R4",
            "mfx_R4",
            "cx_R4",
        ]:
            self._base.out_vars[k] = self._base.in_vars["data_vars"][k]

        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        self.stencil_factory = stencil_factory
        self.namelist = DynamicalCoreConfig.from_namelist(namelist)
        self.grid = grid

        self._are_tracers_setup = False

        self._tracers = None

    def compute_sequential(self, inputs_list, communicator_list):
        print("No serial test available")

    def state_from_inputs(self, inputs: dict, tracers: TracersType) -> SimpleNamespace:
        input_storages = super().state_from_inputs(inputs)
        # Rename fluxes and courant numbers
        input_storages["mfx"] = input_storages.pop("mfx_R4")
        input_storages["mfy"] = input_storages.pop("mfy_R4")
        input_storages["cx"] = input_storages.pop("cx_R4")
        input_storages["cy"] = input_storages.pop("cy_R4")
        # Make tracers
        input_storages["tracers"] = tracers
        return SimpleNamespace(**input_storages)

    def outputs_from_state(self, state: dict):
        if len(self.outputs) == 0:
            return {}
        outputs = {}
        storages = {}
        for name, _properties in self.outputs.items():
            if name in ["mfx_R4", "mfy_R4", "cx_R4", "cy_R4"]:
                storages[name] = state[name[:-3]]
            elif isinstance(state[name], Quantity):
                storages[name] = state[name].data
            elif len(self.outputs[name]["dims"]) > 0:
                storages[name] = state[name]  # assume it's a storage
            else:
                outputs[name] = state[name]  # scalar
        # Put tracers
        storages["tracers"] = state["tracers"].quantity.data[:-1, :-1, :-1, :]
        outputs.update(self._base.slice_output(storages))
        return outputs

    def compute_parallel(self, inputs, communicator):
        # tracers_mapping = Tracers.blind_mapping_from_data(inputs["tracers"])
        # tracers_mapping[0] = "vapor"
        # tracers_mapping[1] = "liquid"
        # tracers_mapping[2] = "rain"
        # tracers_mapping[3] = "snow"
        # tracers_mapping[4] = "ice"
        # tracers_mapping[5] = "graupel"
        # tracers_mapping[6] = "cloud"
        # tracers = Tracers.make_from_4D_array(
        #     self.quantity_factory,
        #     tracers_mapping[0:7],
        #     inputs["tracers"],
        # )

        if not self._are_tracers_setup:
            self._are_tracers_setup = True
            self._tracers = setup_tracers(
                number_of_tracers=inputs["tracers"].shape[3],
                quantity_factory=self.quantity_factory,
                mappings={
                    "vapor": 0,
                    "liquid": 1,
                    "rain": 3,
                    "snow": 4,
                    "ice": 2,
                    "graupel": 5,
                    "cloud": 6,
                },
            )

        self._tracers.quantity.data[:-1, :-1, :-1, :] = inputs["tracers"]

        inputs["te_2d"] = inputs["te_2d"].astype(Float)
        state = self.state_from_inputs(inputs, self._tracers)

        l_to_e = LagrangianToEulerian_GEOS(
            self.stencil_factory,
            self.quantity_factory,
            DynamicalCoreConfig.from_namelist(self.namelist).remapping,
            communicator,
            self.grid.grid_data,
            state.nq,
            state.pfull,
            state.tracers,
            DynamicalCoreConfig.adiabatic,
        )

        l_to_e(
            state.tracers,
            state.pt,
            state.delp,
            state.delz,
            state.peln,
            state.u,
            state.v,
            state.w,
            state.mfx,
            state.mfy,
            state.cx,
            state.cy,
            state.cappa,
            state.q_con,
            state.pkz,
            state.pk,
            state.pe,
            state.phis,
            state.te_2d,
            state.ps,
            state.wsd,
            state.ak,
            state.bk,
            state.dp1,
            state.ptop,
            state.akap,
            state.zvir,
            state.last_step,
            state.consv_te,
            state.mdt,
        )

        outputs = self.outputs_from_state(vars(state))
        return outputs
