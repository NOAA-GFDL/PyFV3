from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl import StencilFactory
from ndsl.dsl.typing import FloatField, Float
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.mapn_tracer import MapNTracer


class TranslateMapN_Tracer_2d(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "qtracers": {
            },
            
            

        }
        # self.in_vars["parameters"] = [
        #     "q_min",
        # ]

        self.out_vars = {
            "qtracers": {
            },
        }

        # Value from GEOS
        self.kord = 9 

        # mode / iv set to 1 from GEOS
        self.mode = 1 

        nq = 9

        fill = True

        # self._mapn_tracer = MapNTracer(
        #     self.stencil_factory,
        #     self.quantity_factory,
        #     abs(self.kord),
        #     nq,
        #     fill=fill,
        #     tracers=tracers,
        # )

    def compute_from_storage(self, inputs):
        # self._compute_func(
        #         inputs["qs_"],
        #         inputs["q4_1"],
        #         inputs["q4_2"],
        #         inputs["q4_3"],
        #         inputs["q4_4"],
        #         inputs["dp1_"],
        #         Float(inputs["q_min"]),
        #     )

        print("qtracers shape: ", inputs["qtracers"].data.shape)
        return inputs
