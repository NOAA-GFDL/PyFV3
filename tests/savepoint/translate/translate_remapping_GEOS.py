from types import SimpleNamespace

from ndsl import Namelist, StencilFactory
from ndsl.constants import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM
from ndsl.dsl.typing import Float
from ndsl.stencils.testing import Grid, ParallelTranslateBaseSlicing
from pyFV3 import DynamicalCoreConfig
from pyFV3.stencils import moist_cv
from pyFV3.stencils.map_single import MapSingle
from pyFV3.stencils.mapn_tracer import MapNTracer
from pyFV3.stencils.mpp_global_sum import mpp_global_sum
from pyFV3.stencils.remapping import (
    init_pe,
    moist_cv_pt_pressure,
    pe0_ptop_xmax,
    pe_pk_delp_peln,
    pn2_pk_delp,
    pressures_mapu,
    pressures_mapv,
)
from pyFV3.stencils.scale_delz import rescale_delz_1, rescale_delz_2
from pyFV3.stencils.w_fix_consrv_moment import W_fix_consrv_moment


class TranslateRemapping_GEOS(ParallelTranslateBaseSlicing):
    inputs = {
        "pe_": {
            "name": "pe_",
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
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "ak": {
            "name": "ak",
            "dims": [X_DIM],
            "units": "No Units",
        },
        "bk": {
            "name": "bk",
            "dims": [X_DIM],
            "units": "No Units",
        },
        "pk": {
            "name": "pk",
            "dims": [X_DIM, Y_DIM, Z_DIM],
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
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "v": {
            "name": "v",
            "dims": [X_DIM, Y_DIM, Z_DIM],
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
        "cosa_s": {
            "name": "cosa_s",
            "dims": [X_DIM, Y_DIM],
            "units": "No Units",
        },
        "rsin2": {
            "name": "rsin2",
            "dims": [X_DIM, Y_DIM],
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
        "area_64_": {
            "name": "area_64_",
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
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "v": {
            "name": "v",
            "dims": [X_DIM, Y_DIM, Z_DIM],
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
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "pe_": {
            "name": "pe_",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "No Units",
        },
        "pk": {
            "name": "pk",
            "dims": [X_DIM, Y_DIM, Z_DIM],
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
            "pe_": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "qvapor": {
                "kend": grid.npz - 1,
            },
            "qliquid": {
                "kend": grid.npz - 1,
            },
            "qice": {
                "kend": grid.npz - 1,
            },
            "qrain": {
                "kend": grid.npz - 1,
            },
            "qsnow": {
                "kend": grid.npz - 1,
            },
            "qgraupel": {
                "kend": grid.npz - 1,
            },
            "qcld": {
                "kend": grid.npz - 1,
            },
            "qo3mr": {
                "kend": grid.npz - 1,
            },
            "qsgs_tke": {
                "kend": grid.npz - 1,
            },
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "cappa": {},
            "ps": {},
            "peln_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "ak": {},
            "bk": {},
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "w": {
                "kend": grid.npz - 1,
            },
            "ws_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "u": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
                "kend": grid.npz,
            },
            "v": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz,
            },
            "mfy": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "cy": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "mfx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "cx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
            "cosa_s": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "rsin2": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "hs": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "te0_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "area_64_": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
        }
        self._base.in_vars["parameters"] = [
            "ptop",
            "r_vir",
            "akap",
            "t_min",
            "kord_wz",
            "w_max",
            "w_min",
            "kord_mt",
            "grav",
            "last_step",
            "do_adiabatic_init",
            "consv",
            "consv_min",
            "cv_air",
            "adiabatic",
        ]
        self._base.out_vars = {
            "pt": {},
            "cappa": {},
            "q_con": {},
            "delp": {},
            "delz": {},
            "qvapor": {
                "kend": grid.npz - 1,
            },
            "qliquid": {
                "kend": grid.npz - 1,
            },
            "qice": {
                "kend": grid.npz - 1,
            },
            "qrain": {
                "kend": grid.npz - 1,
            },
            "qsnow": {
                "kend": grid.npz - 1,
            },
            "qgraupel": {
                "kend": grid.npz - 1,
            },
            "qcld": {
                "kend": grid.npz - 1,
            },
            "qo3mr": {
                "kend": grid.npz - 1,
            },
            "qsgs_tke": {
                "kend": grid.npz - 1,
            },
            "w": {
                "kend": grid.npz - 1,
            },
            "u": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed + 1,
                "kend": grid.npz - 1,
            },
            "v": {
                "istart": grid.isd,
                "iend": grid.ied + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
            "mfy": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "cy": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je + 1,
                "kend": grid.npz - 1,
            },
            "mfx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
            "cx_": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz - 1,
            },
            "peln_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pe_": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
        }

        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # self.namelist found in TranslateDycoreFortranData2Py
        # self.namelist = DynamicalCoreConfig.from_namelist(namelist)
        config = DynamicalCoreConfig.from_namelist(self.namelist).remapping

        hydrostatic = config.hydrostatic
        if hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")

        grid_indexing = stencil_factory.grid_indexing
        self._domain_jextra = (
            grid_indexing.domain[0],
            grid_indexing.domain[1] + 1,
            grid_indexing.domain[2] + 1,
        )

        # Value from GEOS
        self.kord = 9

        # Value from GEOS
        self._kord_tm = 9

        # mode / iv set to 1 from GEOS
        self.mode = 1

        self.nq = 9

        self.fill = True

        self._gz = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=Float,
        )

        self._w2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz,
            ),
            dtype=Float,
        )

        self._zsum1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=Float,
        )

        self._compute_performed = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=bool,
        )

        self._ps = self._pe1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=Float,
        )

        self._pe0 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._pe1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._pe2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._pe3 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._pn1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._pn2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._dp2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._pk2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._phis = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz + 1,
            ),
            dtype=Float,
        )

        self._te_2d = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ),
            dtype=Float,
        )

        self._init_pe = stencil_factory.from_origin_domain(
            init_pe,
            origin=grid_indexing.origin_compute(),
            domain=(grid.nic, grid.njc, grid.npz + 1),
        )

        self._moist_cv_pt_pressure = stencil_factory.from_origin_domain(
            moist_cv_pt_pressure,
            # externals={"kord_tm": config.kord_tm, "hydrostatic": hydrostatic},
            externals={"hydrostatic": hydrostatic},
            origin=grid_indexing.origin_compute(),
            # domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),  # Note : Many intervals go from (0,-1) in this stencil
        )

        self._pn2_pk_delp = stencil_factory.from_origin_domain(
            pn2_pk_delp,
            origin=grid_indexing.origin_compute(add=(0, 0, 1)),
            domain=(grid.nic, grid.njc, grid.npz - 1),
        )

        self._map_scalar = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            self._kord_tm,
            self.mode,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._rescale_delz_1 = stencil_factory.from_origin_domain(
            rescale_delz_1,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self._rescale_delz_2 = stencil_factory.from_origin_domain(
            rescale_delz_2,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self._w_fix_consrv_moment = stencil_factory.from_origin_domain(
            func=W_fix_consrv_moment,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self._pressures_mapu = stencil_factory.from_origin_domain(
            pressures_mapu,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1] + 1,
                grid_indexing.domain[2] + 1,
            ),
        )

        self._pe0_ptop_xmax = stencil_factory.from_origin_domain(
            pe0_ptop_xmax,
            origin=(
                grid_indexing.n_halo + grid_indexing.domain[0],
                grid_indexing.n_halo,
                0,
            ),
            domain=(1, grid_indexing.domain[1] + 1, 1),
        )

        self._pressures_mapv = stencil_factory.from_origin_domain(
            pressures_mapv,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0] + 1,
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),
        )

        self._pe_pk_delp_peln = stencil_factory.from_origin_domain(
            pe_pk_delp_peln,
            origin=grid_indexing.origin_compute(),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                grid_indexing.domain[2] + 1,
            ),
        )

        self._moist_cv_pkz = stencil_factory.from_origin_domain(
            moist_cv.moist_pkz,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self._moist_cv_te = stencil_factory.from_origin_domain(
            moist_cv.moist_te,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz + 1),
        )

        self._te_zsum = stencil_factory.from_origin_domain(
            moist_cv.te_zsum,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self._most_cv_pt_last_step = stencil_factory.from_origin_domain(
            moist_cv.moist_pt_last_step,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self._fill_cond = stencil_factory.from_origin_domain(
            moist_cv.cond_output,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

    def compute_sequential(self, inputs_list, communicator_list):
        print("No serial test available")

    def compute_parallel(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        state_namespace = SimpleNamespace(**state)

        self._init_pe(
            state_namespace.pe_,
            self._pe1,
            self._pe2,
            state_namespace.ptop,
        )

        self._moist_cv_pt_pressure(
            state_namespace.qvapor,
            state_namespace.qliquid,
            state_namespace.qrain,
            state_namespace.qsnow,
            state_namespace.qice,
            state_namespace.qgraupel,
            state_namespace.q_con,
            state_namespace.pt,
            state_namespace.cappa,
            state_namespace.delp,
            state_namespace.delz,
            state_namespace.pe_,
            self._pe2,
            state_namespace.ak,
            state_namespace.bk,
            self._dp2,
            self._ps,
            self._pn1,
            self._pn2,
            state_namespace.peln_3d,
            True,
            Float(state_namespace.r_vir),
        )

        self._pn2_pk_delp(
            self._pe2,
            self._pn2,
            self._pk2,
            Float(state_namespace.akap),
        )

        self._map_scalar(
            state_namespace.pt,
            self._pn1,
            self._pn2,
            qmin=state_namespace.t_min,
            interp=True,
        )

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

        self._mapn_tracer = MapNTracer(
            self.stencil_factory,
            self.quantity_factory,
            abs(self.kord),
            self.nq,
            fill=self.fill,
            tracers=tracers,
        )

        self._mapn_tracer(self._pe1, self._pe2, self._dp2, tracers)

        self._map_single_w = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            state_namespace.kord_wz,
            -2,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._map_single_delz = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            state_namespace.kord_wz,
            1,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._map_single_w(
            state_namespace.w,
            self._pe1,
            self._pe2,
            qs=state_namespace.ws_,
            interp=False,
        )

        self._rescale_delz_1(
            state_namespace.delz,
            state_namespace.delp,
        )

        self._map_single_delz(state_namespace.delz, self._pe1, self._pe2)

        self._rescale_delz_2(
            state_namespace.delz,
            self._dp2,
        )

        self._w_fix_consrv_moment(
            state_namespace.w,
            self._w2,
            self._dp2,
            self._gz,
            state_namespace.w_max,
            state_namespace.w_min,
            self._compute_performed,
        )

        self._map1_ppm_u = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            state_namespace.kord_mt,
            -1,
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
        )

        self._pressures_mapu(
            state_namespace.pe_,
            state_namespace.ak,
            state_namespace.bk,
            self._pe0,
            self._pe3,
            state_namespace.ptop,
        )

        self._pe0_ptop_xmax(
            self._pe0,
            state_namespace.ptop,
        )

        self._map1_ppm_u(
            state_namespace.u,
            self._pe0,
            self._pe3,
            interp=False,
        )

        self._map1_ppm_u(
            state_namespace.mfy,
            self._pe0,
            self._pe3,
            interp=False,
        )

        self._map1_ppm_u(
            state_namespace.cy,
            self._pe0,
            self._pe3,
            interp=False,
        )

        self._map1_ppm_v = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            state_namespace.kord_mt,
            -1,
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
        )

        self._pressures_mapv(
            state_namespace.pe_,
            state_namespace.ak,
            state_namespace.bk,
            self._pe0,
            self._pe3,
        )

        self._map1_ppm_v(
            state_namespace.v,
            self._pe0,
            self._pe3,
            interp=False,
        )

        self._map1_ppm_v(
            state_namespace.mfx_,
            self._pe0,
            self._pe3,
            interp=False,
        )

        self._map1_ppm_v(
            state_namespace.cx_,
            self._pe0,
            self._pe3,
            interp=False,
        )

        self._pe_pk_delp_peln(
            state_namespace.pe_,
            state_namespace.pk,
            state_namespace.delp,
            state_namespace.peln_3d,
            self._pe2,
            self._pk2,
            self._pn2,
            state_namespace.ak,
            state_namespace.bk,
            state_namespace.akap,
            state_namespace.ptop,
        )

        self._moist_cv_pkz(
            state_namespace.qvapor,
            state_namespace.qliquid,
            state_namespace.qrain,
            state_namespace.qsnow,
            state_namespace.qice,
            state_namespace.qgraupel,
            state_namespace.pkz,
            state_namespace.pt,
            state_namespace.cappa,
            state_namespace.delp,
            state_namespace.delz,
            Float(state_namespace.r_vir),
        )

        if state_namespace.last_step and not state_namespace.do_adiabatic_init:

            if state_namespace.consv > state_namespace.consv_min:

                self._moist_cv_te(
                    state_namespace.qvapor,
                    state_namespace.qliquid,
                    state_namespace.qrain,
                    state_namespace.qsnow,
                    state_namespace.qice,
                    state_namespace.qgraupel,
                    state_namespace.u,
                    state_namespace.v,
                    state_namespace.w,
                    self._te_2d,
                    state_namespace.pt,
                    self._phis,
                    state_namespace.delp,
                    state_namespace.rsin2,
                    state_namespace.cosa_s,
                    state_namespace.hs,
                    state_namespace.delz,
                    state_namespace.grav,
                )

                self._te_zsum(
                    self._te_2d,
                    state_namespace.te0_2d_,
                    state_namespace.delp,
                    state_namespace.pkz,
                    self._zsum1,
                )

                # Note, mpp_global_sum is currently set up for the C24 TBC setup
                inputArray = (
                    self._te_2d.data * state_namespace.area_64_.data[0:-1, 0:-1]
                )
                tesum = mpp_global_sum(
                    inputArray[3:27, 3:27], communicator, self.stencil_factory
                )
                # print("tesum: ", tesum)
                inputArray = self._zsum1 * state_namespace.area_64_.data[0:-1, 0:-1]
                zsum = mpp_global_sum(
                    inputArray[3:27, 3:27], communicator, self.stencil_factory
                )
                # print("zsum: ", zsum)
                dtmp = tesum / (state_namespace.cv_air.data * zsum)
                # print("dtmp: ", dtmp)
        # I ignore the E_flux computation since it's not used elsewhere
        # in our current setup once it's computed

        if state_namespace.last_step and not state_namespace.adiabatic:

            self._most_cv_pt_last_step(
                state_namespace.qvapor,
                state_namespace.qliquid,
                state_namespace.qrain,
                state_namespace.qsnow,
                state_namespace.qice,
                state_namespace.qgraupel,
                state_namespace.pt,
                state_namespace.pkz,
                Float(dtmp),
                state_namespace.r_vir,
            )

            self._fill_cond(
                state_namespace.q_con,
                state_namespace.qliquid,
                state_namespace.qrain,
                state_namespace.qsnow,
                state_namespace.qice,
                state_namespace.qgraupel,
            )

        return self.outputs_from_state(state)
