from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl import StencilFactory
from ndsl.dsl.typing import FloatField, Float
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.remapping import pn2_pk_delp



class testClass:
    """
    Class to test with DaCe orchestration. test class is MoistCVPlusPt_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid,
    ):
        self._pn2_pk_delp = stencil_factory.from_origin_domain(
            func=pn2_pk_delp,
            origin=(3,3,1),
            domain=(24,24,71),
        )

    def __call__(
        self,
        dp2: FloatField,
        delp: FloatField,
        pe2: FloatField,
        pn2: FloatField,
        pk: FloatField,
        akap: Float,
    ):
        self._pn2_pk_delp(
            dp2,
            delp,
            pe2,
            pn2,
            pk,
            akap
        )


class TranslatePN2_PK_DelP(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.compute_func = testClass(self.stencil_factory, self.grid)  # type: ignore
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "pe2_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                     },
            "pn2_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,},
            "pk_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,},

        }
        self.in_vars["parameters"] = [
            "akap",
        ]

        self.out_vars = {
            "pe2_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,},
            "pn2_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,},
            "pk_": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,},
        }
        self._dp2 = self.quantity_factory._numpy.zeros(
            (
                31,
                31,
                73,
            ), dtype=Float,
        )

        self._delp = self.quantity_factory._numpy.zeros(
            (
                31,
                31,
                73,
            ), dtype=Float,
        )


    def compute_from_storage(self, inputs):

        # print("delp shape = ", self._delp.shape)
        # print("inputs[pe2] shape = ", inputs["pe2_"].shape)
        # print("inputs[pe2_][:,3,0] = ", inputs["pe2_"][:,3,0])
        # print('self.grid.is_ = ', self.grid.is_)
        # print('self.grid.ie = ', self.grid.ie)
        # print('self.grid.js = ', self.grid.js)
        # print('self.grid.je = ', self.grid.je)
        # print('self.storage_vars() = ', self.storage_vars())
        # self.make_storage_data_input_vars(inputs)
        # exit(1)
        self.compute_func(self._dp2,
                     self._delp,
                     inputs["pe2_"],
                     inputs["pn2_"],
                     inputs["pk_"],
                     inputs["akap"])
        return inputs
