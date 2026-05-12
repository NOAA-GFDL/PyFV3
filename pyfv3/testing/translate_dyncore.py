from f90nml import Namelist

import ndsl.dsl.gt4py_utils as utils
from ndsl import Quantity, StencilFactory
from ndsl.constants import I_DIM, I_INTERFACE_DIM, J_DIM, J_INTERFACE_DIM, K_DIM
from ndsl.stencils.testing import Grid, ParallelTranslate2PyState
from ndsl.typing import Communicator
from pyfv3._config import DynamicalCoreConfig
from pyfv3.dycore_state import DycoreState
from pyfv3.stencils import dyn_core
from pyfv3.tracers import default_GEOS_tracers


class TranslateDynCore(ParallelTranslate2PyState):
    inputs = {
        "q_con": {
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "default",
        },
        "cappa": {
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "default",
        },
        "delp": {
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "default",
        },
        "pt": {
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "K",
        },
        "u": {
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "uc": {
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
        "vc": {
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "m/s",
        },
        "w": {
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "m/s",
        },
    }

    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ) -> None:
        super().__init__(grid, namelist, stencil_factory)
        self._base.in_vars["data_vars"] = {
            "cappa": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "delp": {},
            "pt": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "phis": {"kstart": 0, "kend": 0},
            "wsd": grid.compute_dict(),
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "pkz": grid.compute_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "q_con": {},
            "ak": {},
            "bk": {},
            "diss_estd": {},
            "dpx": grid.compute_dict(),
        }
        self._base.in_vars["data_vars"]["wsd"]["kstart"] = grid.npz
        self._base.in_vars["data_vars"]["wsd"]["kend"] = None

        self._base.in_vars["parameters"] = ["mdt", "akap", "ptop", "n_map"]

        self._base.out_vars = {}
        for v, d in self._base.in_vars["data_vars"].items():
            self._base.out_vars[v] = d

        del self._base.out_vars["ak"]
        del self._base.out_vars["bk"]
        del self._base.out_vars["phis"]
        del self._base.out_vars["pkz"]

        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-6
        self.ignore_near_zero_errors["wsd"] = 1e-18
        self.stencil_factory = stencil_factory
        self.config = DynamicalCoreConfig.from_f90nml(namelist)

    def compute_parallel(self, inputs: dict, communicator: Communicator) -> dict:
        default_GEOS_tracers(self.grid.quantity_factory)
        # ak, bk, and phis are numpy arrays at this point and
        #   must be converted into gt4py storages
        for name in ("ak", "bk", "phis"):
            inputs[name] = utils.make_storage_data(
                inputs[name],
                inputs[name].shape,
                len(inputs[name].shape) * (0,),
                backend=self.stencil_factory.backend,
            )

        grid_data = self.grid.grid_data
        if grid_data.ak is None or grid_data.bk is None:
            grid_data.ak = inputs["ak"]
            grid_data.bk = inputs["bk"]
            grid_data.ptop = inputs["ptop"]
        self._base.make_storage_data_input_vars(inputs)
        inputs_dtypes = {}
        for k, v in inputs.items():
            if hasattr(v, "dtype"):
                inputs_dtypes[k] = v.dtype
        state = DycoreState.init_zeros(
            quantity_factory=self.grid.quantity_factory,
            dtype_dict=inputs_dtypes,
            allow_mismatch_float_precision=True,
        )
        wsd = self.grid.quantity_factory.zeros(
            dims=[I_DIM, J_DIM],
            units="unknown",
        )
        for name, value in inputs.items():
            if hasattr(state, name) and isinstance(state[name], Quantity):
                # the ndarray can have buffer points at the end, so value.shape
                # is often not equal to state[name].shape
                selection = tuple(slice(0, end) for end in value.shape)
                state[name].data[selection] = value
            else:
                setattr(state, name, value)
        phis = self.grid.quantity_factory.zeros(
            dims=[I_DIM, J_DIM],
            units="m",
        )
        phis.data[:] = phis.np.asarray(inputs["phis"])
        dpx = self.grid.quantity_factory.zeros(
            dims=[I_DIM, J_DIM, K_DIM],
            units="unknown",
            dtype=inputs_dtypes["dpx"],
            allow_mismatch_float_precision=True,
        )
        dpx.data[:] = dpx.np.asarray(inputs["dpx"])
        acoustic_dynamics = dyn_core.AcousticDynamics(
            comm=communicator,
            stencil_factory=self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            grid_data=grid_data,
            damping_coefficients=self.grid.damping_coefficients,
            grid_type=self.grid.grid_type,
            nested=self.grid.nested,
            stretched_grid=self.grid.stretched_grid,
            config=self.config.acoustic_dynamics,
            phis=phis,
            state=state,
        )
        acoustic_dynamics.cappa.data[:] = inputs["cappa"][:]

        acoustic_dynamics(
            state,
            mfxd=state.mfxd,
            mfyd=state.mfyd,
            cxd=state.cxd,
            cyd=state.cyd,
            dpx=dpx,
            wsd=wsd,
            timestep=inputs["mdt"],
            n_map=inputs["n_map"],
        )
        # the "inputs" dict is not used to return, we construct a new dict based
        # on variables attached to `state`
        storages_only = {}
        for name, value in vars(state).items():
            if isinstance(value, Quantity):
                storages_only[name] = value[:]
            else:
                storages_only[name] = value
        storages_only["wsd"] = wsd[:]
        storages_only["cappa"] = acoustic_dynamics.cappa[:]
        storages_only["dpx"] = dpx[:]
        return self._base.slice_output(storages_only)
