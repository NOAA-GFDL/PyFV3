import ndsl.dsl.gt4py_utils as utils
from ndsl import Namelist, StencilFactory
from pyFV3 import DynamicalCoreConfig
from pyFV3.stencils.remapping import init_pe, moist_cv_pt_pressure, pn2_pk_delp
from pyFV3.stencils.map_single import MapSingle
from pyFV3.stencils import moist_cv
from pyFV3.stencils.scale_delz import rescale_delz_1, rescale_delz_2
from pyFV3.stencils.w_fix_consrv_moment import W_fix_consrv_moment
from pyFV3.stencils.remapping import pressures_mapu, pe0_ptop_xmax
from ndsl.stencils.testing import pad_field_in_j, Grid
from pyFV3.testing import TranslateDycoreFortranData2Py
from ndsl.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from ndsl.dsl.typing import Float, FloatField
from pyFV3.stencils.mapn_tracer import MapNTracer

def moist_pt(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: Float,
):
    with computation(PARALLEL), interval(...):
        cvm, gz, q_con, cappa, pt = moist_cv.moist_pt_func(
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            q_con,
            pt,
            cappa,
            delp,
            delz,
            r_vir,
        )

class TranslateRemapping_GEOS(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid: Grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        
        self.in_vars["data_vars"] = {
            
            "pe_": {
                "istart": grid.is_-1,
                "iend": grid.ie+1,
                "jstart": grid.js-1,
                "jend": grid.je+1,
                "kend": grid.npz + 1,
            },
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz
            },
            "qvapor": {"serialname": "qvapor_js"},
            "qliquid": {"serialname": "qliquid_js"},
            "qice": {"serialname": "qice_js"},
            "qrain": {"serialname": "qrain_js"},
            "qsnow": {"serialname": "qsnow_js"},
            "qgraupel": {"serialname": "qgraupel_js"},
            "qcld": {"serialname": "qcld_js"},
            "qo3mr": {
                "kend": grid.npz-1,
            },
            "qsgs_tke": {
                "kend": grid.npz-1,
            },
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "cappa": {},
            "ps": {},
            "pn1_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pn2_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "peln_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "ak": {},
            "bk": {},
            "dp2_3d": { 
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pk2_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "w": {
                "kend": grid.npz-1,
                },
            "ws_":{
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "u": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed+1,
                "kend": grid.npz-1,
            },
            "mfy": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz-1,
            },

            "cy": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz-1,
            },
        }
        # self.write_vars = ["gz", "cvm"]
        self.write_vars = ["qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel","qcld"]
        for k, v in self.in_vars["data_vars"].items():
            # if k not in self.write_vars:
            if k in self.write_vars:
                v["axis"] = 1
        self.in_vars["parameters"] = [
            "ptop",
            "r_vir",
            # "remap_t", # For some reason, translate test can't accept a logical variable
            "akap",
            "t_min",
            "kord_wz",
            "w_max",
            "w_min",
            "kord_mt",
            # "zvir",
            # "last_step",
            # "consv_te",
            # "mdt",
            # "nq",
        ]
        self.out_vars = {
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pe0_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz
            },
            "pe3_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz
            },
            "pt": {},
            "cappa": {},
            "q_con": {},
            "delp": {},
            "delz": {},
            "ps": {},
            "dp2_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "pn1_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pn2_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "pk2_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },

            "qvapor": {
                "kend": grid.npz-1,
            },
            "qliquid": {
                "kend": grid.npz-1,
                },
            "qice": {
                "kend": grid.npz-1,
            },
            "qrain": {
                "kend": grid.npz-1,
            },
            "qsnow": {
                "kend": grid.npz-1,
            },
            "qgraupel": {
                "kend": grid.npz-1,
            },
            "qcld": {
                "kend": grid.npz-1,
            },
            "qo3mr": {
                "kend": grid.npz-1,
            },
            "qsgs_tke": {
                "kend": grid.npz-1,
            },
            "w": {
                "kend": grid.npz-1,
            },
            "u": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed+1,
                "kend": grid.npz-1,
            },

            "mfy": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz-1,
            },

            "cy": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.js,
                "jend": grid.je+1,
                "kend": grid.npz-1,
            },
        }

        self.stencil_factory = stencil_factory
        self.quantity_factory = grid.quantity_factory

        # self.namelist found in TranslateDycoreFortranData2Py
        # self.namelist = DynamicalCoreConfig.from_namelist(namelist)
        config=DynamicalCoreConfig.from_namelist(self.namelist).remapping

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
            ), dtype=Float,
        )

        self._w2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz,
            ), dtype=Float,
        )

        self._compute_performed = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ), dtype=bool,
        )

        self._init_pe = stencil_factory.from_origin_domain(
            init_pe, 
            origin=grid_indexing.origin_compute(),
            domain=(grid_indexing.domain[0],1,grid_indexing.domain[2] + 1),
        )

        self._moist_cv_pt = stencil_factory.from_origin_domain(
            moist_pt,
            origin=grid.compute_origin(),
            domain=(grid_indexing.domain[0], 1, grid_indexing.domain[2]),
        )

        self._moist_cv_pt_pressure = stencil_factory.from_origin_domain(
            moist_cv_pt_pressure,
            # externals={"kord_tm": config.kord_tm, "hydrostatic": hydrostatic},
            externals={"hydrostatic": hydrostatic},
            origin=grid_indexing.origin_compute(),
            # domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            domain=(grid_indexing.domain[0], 1, grid_indexing.domain[2]+1), # Note : Many intervals go from (0,-1) in this stencil
        )

        self._pn2_pk_delp = stencil_factory.from_origin_domain(
            pn2_pk_delp,
            origin=grid_indexing.origin_compute(add=(0,0,1)),
            domain=(grid.nic, 1, grid.npz-1),
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
            domain=(grid.nic, 1, grid.npz),
        )

        self._rescale_delz_2 = stencil_factory.from_origin_domain(
            rescale_delz_2,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

        self._w_fix_consrv_moment = stencil_factory.from_origin_domain(
            func=W_fix_consrv_moment,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

        self._pressures_mapu = stencil_factory.from_origin_domain(
            pressures_mapu,
            origin=grid_indexing.origin_compute(),
            domain=(grid_indexing.domain[0],1,grid_indexing.domain[2] + 1)
        )

        self._pe0_ptop_xmax = stencil_factory.from_origin_domain(
            pe0_ptop_xmax,
            origin=(grid_indexing.domain[0]+3,3,0),
            domain=(1,1,grid_indexing.domain[2] + 1)
        )

        # self._pressures_mapv = stencil_factory.from_origin_domain(
        #     pressures_mapv,
        #     origin=grid_indexing.origin_compute(),
        #     domain=(
        #         grid_indexing.domain[0] + 1,
        #         grid_indexing.domain[1],
        #         grid_indexing.domain[2] + 1,
        #     ),
        # )

        # self._map1_ppm_v = MapSingle(
        #     self.stencil_factory,
        #     self.quantity_factory,
        #     inputs["kord_mt"],
        #     -1,
        #     dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
        # )

        # self._pe_pk_delp_peln = stencil_factory.from_origin_domain(
        #     pe_pk_delp_peln,
        #     origin=grid_indexing.origin_compute(),
        #     domain=self._domain_kextra,
        # )

    def compute_from_storage(self, inputs):

        # Replicates tracer values in I along the J direction
        for name, value in inputs.items():
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )

        self._init_pe(
            inputs["pe_"],
            inputs["pe1_"],
            inputs["pe2_"],
            inputs["ptop"],
        )

        self._moist_cv_pt_pressure(
            inputs["qvapor"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
            inputs["q_con"],
            inputs["pt"],
            inputs["cappa"],
            inputs["delp"],
            inputs["delz"],
            inputs["pe_"],
            inputs["pe2_"],
            inputs["ak"],
            inputs["bk"],
            inputs["dp2_3d"],
            inputs["ps"],
            inputs["pn1_3d"],
            inputs["pn2_3d"],
            inputs["peln_3d"],
            True,
            Float(inputs["r_vir"]),
        )

        self._pn2_pk_delp(
            inputs["pe2_"],
            inputs["pn2_3d"],
            inputs["pk2_3d"],
            Float(inputs["akap"]),
        )

        self._map_scalar(
                inputs["pt"],
                inputs["pn1_3d"],
                inputs["pn2_3d"],
                qmin=inputs["t_min"],
                interp=True,
        )

        tracers = { "qvapor": inputs["qvapor"],
                    "qliquid": inputs["qliquid"],
                    "qice": inputs["qice"],
                    "qrain": inputs["qrain"],
                    "qsnow": inputs["qsnow"],
                    "qgraupel": inputs["qgraupel"],
                    "qcld": inputs["qcld"],
                    "qo3mr": inputs["qo3mr"],
                    "qsgs_tke": inputs["qsgs_tke"],
        }

        self._mapn_tracer = MapNTracer(
            self.stencil_factory,
            self.quantity_factory,
            abs(self.kord),
            self.nq,
            fill=self.fill,
            tracers=tracers,
        )

        self._mapn_tracer(inputs["pe1_"], inputs["pe2_"], inputs["dp2_3d"], tracers)

        self._map_single_w = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_wz"],
            -2,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self._map_single_delz = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_wz"],
            1,
            dims=[X_DIM, Y_DIM, Z_DIM],
        )
        self._map_single_w(inputs["w"], inputs["pe1_"], inputs["pe2_"], qs=inputs["ws_"], interp=False)
        
        self._rescale_delz_1(
            inputs["delz"],
            inputs["delp"],
        )
        
        self._map_single_delz(inputs["delz"], inputs["pe1_"], inputs["pe2_"])

        self._rescale_delz_2(
            inputs["delz"],
            inputs["dp2_3d"],
        )
        
        self._w_fix_consrv_moment(
                     inputs["w"],
                     self._w2,
                     inputs["dp2_3d"],
                     self._gz,
                     inputs["w_max"],
                     inputs["w_min"],
                     self._compute_performed
                     )

        self._map1_ppm_u = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_mt"],
            -1,
            dims=[X_DIM, Y_INTERFACE_DIM, Z_DIM],
        )

        self._pressures_mapu(
                inputs["pe_"],
                inputs["ak"],
                inputs["bk"],
                inputs["pe0_"],
                inputs["pe3_"],
                inputs["ptop"],
            )
        
        self._pe0_ptop_xmax(
                inputs["pe0_"],
                inputs["ptop"],
            )

        self._map1_ppm_u(
                inputs["u"],
                inputs["pe0_"],
                inputs["pe3_"],
                interp=False,
            )
        
        self._map1_ppm_u(
                inputs["mfy"],
                inputs["pe0_"],
                inputs["pe3_"],
                interp=False,
            )
        
        self._map1_ppm_u(
                inputs["cy"],
                inputs["pe0_"],
                inputs["pe3_"],
                interp=False,
            )

        # self._pressures_mapv(
        #         inputs["pe_"],
        #         inputs["ak"],
        #         inputs["bk"],
        #         inputs["pe0_v"],
        #         inputs["pe3_v"],
        #     )

        # self._map1_ppm_v(
        #         inputs["v_"],
        #         inputs["pe0_v"],
        #         inputs["pe3_v"],
        #         interp=False,
        #     )
        
        # self._map1_ppm_v(
        #         inputs["mfx_"],
        #         inputs["pe0_v"],
        #         inputs["pe3_v"],
        #         interp=False,
        #     )
        
        # self._map1_ppm_v(
        #         inputs["cx_"],
        #         inputs["pe0_v"],
        #         inputs["pe3_v"],
        #         interp=False,
        #     )

        # self._pe_pk_delp_peln(inputs["pe_"],
        #                       inputs["pk"],
        #                       inputs["delp"],
        #                       inputs["peln_"],
        #                       inputs["pe2_"],
        #                       inputs["pk2_"],
        #                       inputs["pn2_"],
        #                       inputs["ak"],
        #                       inputs["bk"],
        #                       inputs["akap"],
        #                       inputs["ptop"],
        # )

        # NOTE : THERE WILL BE ADJUSTMENTS TO ACCOUNT FOR PKZ
        # self._moist_cv_pt(
        #     inputs["qvapor"],
        #     inputs["qliquid"],
        #     inputs["qrain"],
        #     inputs["qsnow"],
        #     inputs["qice"],
        #     inputs["qgraupel"],
        #     inputs["q_con"],
        #     inputs["pt"],
        #     inputs["cappa"],
        #     inputs["delp"],
        #     inputs["delz"],
        #     inputs["r_vir"],
        # )

        # If loop based on if( last_step .and. (.not.do_adiabatic_init)  ) then
            # PHIS computation
            # Some variation of moist_cv_pt
            # zsum1 computation
        
            # MPP GLOBAL SUM
            # E_FLUX calcuation

        # If loop based on if ( last_step .and. (.not. adiabatic) ) then
            # Some variation of moist_cv_pt
            # Condensation update

        return inputs
