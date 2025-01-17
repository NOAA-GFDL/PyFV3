import ndsl.dsl.gt4py_utils as utils
from ndsl import Namelist, StencilFactory
from pyFV3 import DynamicalCoreConfig
from pyFV3.stencils.remapping import init_pe, moist_cv_pt_pressure, pn2_pk_delp
from pyFV3.stencils.map_single import MapSingle
from pyFV3.stencils import moist_cv
from pyFV3.stencils.scale_delz import rescale_delz_1, rescale_delz_2
from pyFV3.stencils.w_fix_consrv_moment import W_fix_consrv_moment
from pyFV3.stencils.remapping import pressures_mapu, pe0_ptop_xmax, pressures_mapv, pe_pk_delp_peln
from pyFV3.stencils.mpp_global_sum import mpp_global_sum
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
                "kend": grid.npz,
            },
            "v": {
                "istart": grid.isd,
                "iend": grid.ied+1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz,
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
            "mfx_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },

            "cx_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz-1,
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
            "te_2d_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
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
            "area_64": {
                "istart": grid.isd,
                "iend": grid.ied,
                "jstart": grid.jsd,
                "jend": grid.jed,
            },
            "te": {}
        }
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
            "grav",
            # "zvir",
            # "last_step",
            # "consv_te",
            # "mdt",
            # "nq",
        ]
        self.out_vars = {
            # "pe1_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "pe2_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "pe0_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie+1,
            #     "jstart": grid.js,
            #     "jend": grid.je+1,
            #     "kend": grid.npz+1
            # },
            # "pe3_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie+1,
            #     "jstart": grid.js,
            #     "jend": grid.je+1,
            #     "kend": grid.npz+1
            # },
            "pt": {},
            "cappa": {},
            # "q_con": {},
            "delp": {},
            "delz": {},
            # "ps": {},
            # "dp2_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz-1,
            # },
            # "pn1_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "pn2_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "pk2_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },

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
            "v": {
                "istart": grid.isd,
                "iend": grid.ied+1,
                "jstart": grid.jsd,
                "jend": grid.jed,
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
            "mfx_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },

            "cx_": {
                "istart": grid.is_,
                "iend": grid.ie+1,
                "jstart": grid.jsd,
                "jend": grid.jed,
                "kend": grid.npz-1,
            },

            "peln_3d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },

            "pe_": {
                "istart": grid.is_-1,
                "iend": grid.ie+1,
                "jstart": grid.js-1,
                "jend": grid.je+1,
                "kend": grid.npz + 1,
            },

            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            # "te_2d_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            # },
            # "te": {}
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

        self._zsum1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ), dtype=Float,
        )

        self._compute_performed = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ), dtype=bool,
        )

        self._ps = self._pe1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ), dtype=Float,
        )

        self._pe0 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._pe1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._pe2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._pe3 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._pn1 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._pn2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._dp2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        
        )

        self._pk2 = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
        )

        self._phis = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz+1,
            ), dtype=Float,
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
            domain=(grid_indexing.domain[0], grid_indexing.domain[1], grid_indexing.domain[2]+1), # Note : Many intervals go from (0,-1) in this stencil
        )

        self._pn2_pk_delp = stencil_factory.from_origin_domain(
            pn2_pk_delp,
            origin=grid_indexing.origin_compute(add=(0,0,1)),
            domain=(grid.nic, grid.njc, grid.npz-1),
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
            domain=(grid_indexing.domain[0],grid_indexing.domain[1],grid_indexing.domain[2] + 1)
        )

        self._pe0_ptop_xmax = stencil_factory.from_origin_domain(
            pe0_ptop_xmax,
            origin=(grid_indexing.n_halo+grid_indexing.domain[0],grid_indexing.n_halo,0),
            domain=(1,grid_indexing.domain[1], 1)
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
            domain=(grid_indexing.domain[0], grid_indexing.domain[1], grid_indexing.domain[2] + 1,
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
            domain=(grid.nic, 1, grid.npz+1),
        )

        self._te_zsum = stencil_factory.from_origin_domain(
            moist_cv.te_zsum,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

        self._most_cv_pt_last_step = stencil_factory.from_origin_domain(
            moist_cv.moist_pt_last_step,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

    def compute_from_storage(self, inputs):

        self._init_pe(
            inputs["pe_"],
            self._pe1,
            self._pe2,
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
            self._pe2,
            inputs["ak"],
            inputs["bk"],
            self._dp2,
            self._ps,
            self._pn1,
            self._pn2,
            inputs["peln_3d"],
            True,
            Float(inputs["r_vir"]),
        )

        self._pn2_pk_delp(
            self._pe2,
            self._pn2,
            self._pk2,
            Float(inputs["akap"]),
        )

        self._map_scalar(
                inputs["pt"],
                self._pn1,
                self._pn2,
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

        self._mapn_tracer(self._pe1, 
                         self._pe2, 
                         self._dp2,
                         tracers)

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
        self._map_single_w(inputs["w"], 
                           self._pe1, 
                           self._pe2, 
                           qs=inputs["ws_"], 
                           interp=False)
        
        self._rescale_delz_1(
            inputs["delz"],
            inputs["delp"],
        )
        
        self._map_single_delz(inputs["delz"], 
                              self._pe1, 
                              self._pe2)

        self._rescale_delz_2(
            inputs["delz"],
            self._dp2,
        )
        
        self._w_fix_consrv_moment(
                     inputs["w"],
                     self._w2,
                     self._dp2,
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
                self._pe0,
                self._pe3,
                inputs["ptop"],
            )
        
        self._pe0_ptop_xmax(
                self._pe0,
                inputs["ptop"],
            )

        self._map1_ppm_u(
                inputs["u"],
                self._pe0,
                self._pe3,
                interp=False,
            )
        
        self._map1_ppm_u(
                inputs["mfy"],
                self._pe0,
                self._pe3,
                interp=False,
            )
        
        self._map1_ppm_u(
                inputs["cy"],
                self._pe0,
                self._pe3,
                interp=False,
            )
        
        self._map1_ppm_v = MapSingle(
            self.stencil_factory,
            self.quantity_factory,
            inputs["kord_mt"],
            -1,
            dims=[X_INTERFACE_DIM, Y_DIM, Z_DIM],
        )

        self._pressures_mapv(
                inputs["pe_"],
                inputs["ak"],
                inputs["bk"],
                self._pe0,
                self._pe3,
            )

        self._map1_ppm_v(
                inputs["v"],
                self._pe0,
                self._pe3,
                interp=False,
            )
        
        self._map1_ppm_v(
                inputs["mfx_"],
                self._pe0,
                self._pe3,
                interp=False,
            )
        
        self._map1_ppm_v(
                inputs["cx_"],
                self._pe0,
                self._pe3,
                interp=False,
            )

        self._pe_pk_delp_peln(inputs["pe_"],
                              inputs["pk"],
                              inputs["delp"],
                              inputs["peln_3d"],
                              self._pe2,
                              self._pk2,
                              self._pn2,
                              inputs["ak"],
                              inputs["bk"],
                              inputs["akap"],
                              inputs["ptop"],
        )

        self._moist_cv_pkz(
            inputs["qvapor"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
            inputs["pkz"],
            inputs["pt"],
            inputs["cappa"],
            inputs["delp"],
            inputs["delz"],
            Float(inputs["r_vir"]),
        )

        # # May need if loop here based on if( last_step .and. (.not.do_adiabatic_init)  ) then

        # self._moist_cv_te(inputs["qvapor"],
        #                   inputs["qliquid"],
        #                   inputs["qrain"],
        #                   inputs["qsnow"],
        #                   inputs["qice"],
        #                   inputs["qgraupel"],
        #                   inputs["u"],
        #                   inputs["v"],
        #                   inputs["w"],
        #                   inputs["te_2d_"],
        #                   inputs["pt"],
        #                   inputs["phis_"],
        #                   inputs["delp"],
        #                   inputs["rsin2"],
        #                   inputs["cosa_s"],
        #                   inputs["hs"],
        #                   inputs["delz"],
        #                   inputs["grav"],
        #                 )

        # self._te_zsum(inputs["te_2d_"],
        #               inputs["te0_2d_"],
        #               inputs["delp"],
        #               inputs["pkz"],
        #               self._zsum1,
        #             )
        
        # # Note, since this is a serial translate test, mpp_global_sum won't work without the communicator.
        # # Also, mpp_global_sum is currently set up for the C24 TBC setup

        # # tesum = mpp_global_sum(inputs["te_2d_"]*inputs["area_64"], communicator, self.stencil_factory)

        # # I ignore the E_flux computation since it's not used elsewhere in our current setup once it's computed

        # # zsum = mpp_global_sum(self._zsum1*inputs["area_64"], communicator, self.stencil_factory)
        # # dtmp = tesum / (cv_air*zsum)

        # # If loop based on if ( last_step .and. (.not. adiabatic) ) then
        # # self._most_cv_pt_last_step(inputs["qvapor"],
        # #                            inputs["qliquid"],
        # #                            inputs["qrain"],
        # #                            inputs["qsnow"],
        # #                            inputs["qice"],
        # #                            inputs["qgraupel"],
        # #                            self._gz,
        # #                            inputs["pt"],
        # #                            inputs["pkz"],
        # #                            dtmp,
        # #                            inputs["r_vir],
        # #                         )

        return inputs
