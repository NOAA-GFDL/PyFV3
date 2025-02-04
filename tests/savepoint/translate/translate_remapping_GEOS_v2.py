from types import SimpleNamespace

from ndsl import Namelist, StencilFactory
from ndsl.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from ndsl.stencils.testing import Grid, ParallelTranslateBaseSlicing
from pyFV3 import DynamicalCoreConfig
from pyFV3.stencils.remapping_GEOS import LagrangianToEulerian_GEOS


# from pyFV3._config import RemappingConfig


class TranslateRemapping_GEOS_v2(ParallelTranslateBaseSlicing):
    inputs = {
        "pe_": {
            "name": "pe_",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "qvapor": {
            "name": "qvapor",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qliquid": {
            "name": "qliquid",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qice": {
            "name": "qice",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qrain": {
            "name": "qrain",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qsnow": {
            "name": "qsnow",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qgraupel": {
            "name": "qgraupel",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qcld": {
            "name": "qcld",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qo3mr": {
            "name": "qo3mr",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qsgs_tke": {
            "name": "qsgs_tke",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "delp": {
            "name": "delp",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "delz": {
            "name": "delz",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "q_con": {
            "name": "q_con",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "pt": {
            "name": "pt",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "cappa": {
            "name": "cappa",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "ps": {
            "name": "ps",
            "dims": [X_DIM, Y_DIM],
            "units": "No Units",
        },
        "peln_3d": {
            "name": "peln_3d",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "ak": {
            "name": "ak",
            "dims": [Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "bk": {
            "name": "bk",
            "dims": [Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "pk": {
            "name": "pk",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "pkz": {
            "name": "pkz",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "w": {
            "name": "w",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "ws_": {
            "name": "ws_",
            "dims": [X_DIM, Y_DIM],
            "units": "No Units",
        },
        "u": {
            "name": "u",
            "dims": [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            "units": "No Units",
        },
        "v": {
            "name": "v",
            "dims": [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "mfy": {
            "name": "mfy",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "cy": {
            "name": "cy",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "mfx_": {
            "name": "mfx_",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "cx_": {
            "name": "cx_",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "hs": {
            "name": "hs",
            "dims": [X_DIM, Y_DIM],
            "units": "No Units",
        },
        "te0_2d_": {
            "name": "te0_2d_",
            "dims": [X_DIM, Y_DIM],
            "units": "No Units",
        },
    }
    outputs = {
        "pt": {
            "name": "pt",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "cappa": {
            "name": "cappa",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "delp": {
            "name": "delp",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "delz": {
            "name": "delz",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qvapor": {
            "name": "qvapor",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qliquid": {
            "name": "qliquid",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qice": {
            "name": "qice",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qrain": {
            "name": "qrain",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qsnow": {
            "name": "qsnow",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qgraupel": {
            "name": "qgraupel",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qcld": {
            "name": "qcld",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qo3mr": {
            "name": "qo3mr",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "qsgs_tke": {
            "name": "qsgs_tke",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "w": {
            "name": "w",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "u": {
            "name": "u",
            "dims": [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            "units": "No Units",
        },
        "v": {
            "name": "v",
            "dims": [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "mfy": {
            "name": "mfy",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "cy": {
            "name": "cy",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "mfx_": {
            "name": "mfx_",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "cx_": {
            "name": "cx_",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "peln_3d": {
            "name": "peln_3d",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "pe_": {
            "name": "pe_",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "pk": {
            "name": "pk",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "No Units",
        },
        "pkz": {
            "name": "pkz",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "q_con": {
            "name": "q_con",
            "dims": [X_DIM, Y_DIM, Z_DIM],
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
                "kend": grid.npz,
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
            "hs": {"serialname": "phis"},
            "ps": {},
            "wsd": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            # column variables...
            "ak": {},
            "bk": {},
            "pfull": {},
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
            "pe",
            "pkz",
            "pk",
            "peln",
            "pt",
            "tracers",
            "cappa",
            "delp",
            "delz",
            "q_con",
            "u",
            "v",
            "w",
            "ps",
            "dp1",
        ]:
            self._base.out_vars[k] = self._base.in_vars["data_vars"][k]

        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        self.stencil_factory = stencil_factory
        self.namelist = DynamicalCoreConfig.from_namelist(namelist)
        self.grid = grid

    def compute_sequential(self, inputs_list, communicator_list):
        print("No serial test available")

    def compute_parallel(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        state_namespace = SimpleNamespace(**state)

        tracers = {
            "qvapor": state_namespace.qvapor,
            "qliquid": state_namespace.qliquid,
            "qice": state_namespace.qice,
            "qrain": state_namespace.qrain,
            "qsnow": state_namespace.qsnow,
            "qgraupel": state_namespace.qgraupel,
            "qcld": state_namespace.qcld,
            "qo3mr": state_namespace.qo3mr,
            "qsgs_tke": state_namespace.qsgs_tke,
        }

        l_to_e = LagrangianToEulerian_GEOS(
            self.stencil_factory,
            self.quantity_factory,
            DynamicalCoreConfig.from_namelist(self.namelist).remapping,
            communicator,
            self.grid.grid_data,
            state_namespace.nq,
            state_namespace.pfull,
            tracers,
            state_namespace.adiabatic,
        )

        l_to_e(
            tracers,
            state_namespace.pt,
            state_namespace.delp,
            state_namespace.delz,
            state_namespace.peln,
            state_namespace.u,
            state_namespace.v,
            state_namespace.w,
            state_namespace.mfx,
            state_namespace.mfy,
            state_namespace.cx,
            state_namespace.cy,
            state_namespace.cappa,
            state_namespace.q_con,
            state_namespace.pkz,
            state_namespace.pk,
            state_namespace.pe,
            state_namespace.hs,
            state_namespace.te0_2d,
            state_namespace.ps,
            state_namespace.wsd,
            state_namespace.ak,
            state_namespace.bk,
            state_namespace.dp1,
            state_namespace.ptop,
            state_namespace.akap,
            state_namespace.zvir,
            state_namespace.last_step,
            state_namespace.consv_te,
            state_namespace.mdt,
        )

        return self.outputs_from_state(state)
