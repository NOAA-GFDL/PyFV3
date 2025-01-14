from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl import StencilFactory
from ndsl.dsl.typing import FloatField, Float
from ndsl.stencils.testing import TranslateFortranData2Py, pad_field_in_j
from pyFV3.stencils import moist_cv


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


class MoistPT:
    """
    Class to test with DaCe orchestration. test class is MoistCVPlusPt_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid,
    ):
        self._moist_cv_pt = stencil_factory.from_origin_domain(
            moist_pt,
            origin=grid.compute_origin(),
            domain=(grid.nic, 1, grid.npz),
        )

    def __call__(
        self,
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
        r_vir: float,
    ):
        self._moist_cv_pt(
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


class TranslateMoistCVPlusPt_2d_last_step(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.compute_func = MoistPT(stencil_factory, self.grid)  # type: ignore
        self.in_vars["data_vars"] = {
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
            "pt": {},
            "pkz": {},
        }
        # self.write_vars = ["qvapor", "qliquid", "qice", "qrain", "qsnow", "qgraupel","qcld"]
        # for k, v in self.in_vars["data_vars"].items():
        #     # if k not in self.write_vars:
        #     if k in self.write_vars:
        #         v["axis"] = 1

        self.in_vars["parameters"] = ["r_vir", "dtmp"]
        self.out_vars = {
            "pt": {},
        }

        self.compute_func = stencil_factory.from_origin_domain(
            moist_cv.moist_pt_last_step,
            origin=grid.compute_origin(),
            domain=(grid.nic, grid.njc, grid.npz),
        )

        self.quantity_factory = grid.quantity_factory

        self._gz = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
                grid.npz,
            ), dtype=Float,
        )

    def compute_from_storage(self, inputs):
        # for name, value in inputs.items():
        #     if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
        #         inputs[name] = self.make_storage_data(
        #             pad_field_in_j(
        #                 value, self.grid.njd, backend=self.stencil_factory.backend
        #             )
        #         )
        self.compute_func(inputs["qvapor"],
                          inputs["qliquid"],
                          inputs["qrain"],
                          inputs["qsnow"],
                          inputs["qice"],
                          inputs["qgraupel"],
                          self._gz,
                          inputs["pt"],
                          inputs["pkz"],
                          inputs["dtmp"],
                          inputs["r_vir"],
                        )
        return inputs
