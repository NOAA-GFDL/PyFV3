import ndsl.dsl.gt4py_utils as utils
from ndsl import Namelist, StencilFactory
from pyFV3 import DynamicalCoreConfig
from pyFV3.stencils.remapping import init_pe, moist_cv_pt_pressure, pn2_pk_delp
from pyFV3.stencils import moist_cv
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
            "qvapor": {"serialname": "qvapor_js"},
            "qliquid": {"serialname": "qliquid_js"},
            "qice": {"serialname": "qice_js"},
            "qrain": {"serialname": "qrain_js"},
            "qsnow": {"serialname": "qsnow_js"},
            "qgraupel": {"serialname": "qgraupel_js"},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "cappa": {},
            # "ps": {},
            # "pn2_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "peln_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "ak": {},
            # "bk": {},
            # "dp2_3d": grid.compute_dict(),
            # "pk": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # }
        }
        self.write_vars = ["gz", "cvm"]
        for k, v in self.in_vars["data_vars"].items():
            if k not in self.write_vars:
                v["axis"] = 1
        self.in_vars["parameters"] = [
            "ptop",
            "r_vir",
            # "remap_t", # For some reason, translate test can't accept a logical variable
            # "akap",
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
            "pt": {},
            "cappa": {},
            "q_con": {},
            # "delp": {},
            # "delz": {},
            # "ps": {},
            # "dp2_3d": grid.compute_dict(),
            # "pn2_3d": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # },
            # "pk": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz + 1,
            # }
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
        
        # self._pe1 = self.quantity_factory.zeros(
        #     [X_DIM, Y_DIM, Z_INTERFACE_DIM],
        #     units="Pa",
        #     dtype=Float,
        # )
        # self._pe2 = self.quantity_factory.zeros(
        #     [X_DIM, Y_DIM, Z_INTERFACE_DIM],
        #     units="Pa",
        #     dtype=Float,
        # )

        self._init_pe = stencil_factory.from_origin_domain(
            init_pe, 
            origin=grid_indexing.origin_compute(), 
            # domain=(grid.nic,1,73),
            domain=(grid_indexing.domain[0],1,grid_indexing.domain[2] + 1),
        )

        self._moist_cv_pt = stencil_factory.from_origin_domain(
            moist_pt,
            origin=grid.compute_origin(),
            domain=(grid_indexing.domain[0], 1, grid_indexing.domain[2]),
        )

        # self._moist_cv_pt_pressure = stencil_factory.from_origin_domain(
        #     moist_cv_pt_pressure,
        #     # externals={"kord_tm": config.kord_tm, "hydrostatic": hydrostatic},
        #     externals={"hydrostatic": hydrostatic},
        #     origin=grid_indexing.origin_compute(),
        #     # domain=grid_indexing.domain_compute(add=(0, 0, 1)),
        #     domain=(grid.nic, 1, grid.npz+1), # Note : Many intervals go from (0,-1) in this stencil
        # )

        # self._pn2_pk_delp = stencil_factory.from_origin_domain(
        #     pn2_pk_delp,
        #     origin=grid_indexing.origin_compute(),
        #     domain=(grid.nic, 1, grid.npz+1),
        # )

        # self._compute_func = MapNTracer(
        #     self.stencil_factory,
        #     self.quantity_factory,
        #     abs(self.kord),
        #     self.nq,
        #     fill=self.fill,
        #     tracers=tracers,
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
                # print("name = ", name)
                # print("value.shape = ", value.shape)
        # print("inputs[qvapor].data.shape() 2 = ", inputs["qvapor"].data.shape)
        self._init_pe(
            inputs["pe_"],
            inputs["pe1_"],
            inputs["pe2_"],
            inputs["ptop"],
        )

        self._moist_cv_pt(
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
            inputs["r_vir"],
        )

        # self._moist_cv_pt_pressure(
        #     inputs["qvapor_"],
        #     inputs["qliquid_"],
        #     inputs["qrain_"],
        #     inputs["qsnow_"],
        #     inputs["qice_"],
        #     inputs["qgraupel_"],
        #     inputs["q_con"],
        #     inputs["pt"],
        #     inputs["cappa"],
        #     inputs["delp"],
        #     inputs["delz"],
        #     inputs["pe_"],
        #     inputs["pe2_"],
        #     inputs["ak"],
        #     inputs["bk"],
        #     inputs["dp2_3d"],
        #     inputs["ps"],
        #     inputs["pn2_3d"],
        #     inputs["peln_3d"],
        #     True,
        #     Float(inputs["r_vir"]),
        # )

        # self._pn2_pk_delp(
        #     inputs["pe2_"],
        #     inputs["pn2_3d"],
        #     inputs["pk"],
        #     Float(inputs["akap"]),
        # )

        # # now that we have the pressure profiles, we can start remapping
        # self._map_single_pt(pt, peln, self._pn2, qmin=self._t_min)

        # self._mapn_tracer(self._pe1, self._pe2, self._dp2, tracers)

        # self._map_single_w(w, self._pe1, self._pe2, qs=wsd)
        # self._map_single_delz(delz, self._pe1, self._pe2)

        return inputs
