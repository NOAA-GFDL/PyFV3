from f90nml import Namelist

from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, I_INTERFACE_DIM, J_DIM, J_INTERFACE_DIM, K_DIM
from ndsl.stencils.testing import ParallelTranslate2Py
from pyfv3.stencils.tracer_2d_1l import TracerCMax


class TranslateTracerCMax(ParallelTranslate2Py):
    inputs = {
        "cx_R4": {
            "name": "cx_R4",
            "dims": [I_INTERFACE_DIM, J_DIM, K_DIM],
            "units": "unitless",
        },
        "cy_R4": {
            "name": "cy_R4",
            "dims": [I_DIM, J_INTERFACE_DIM, K_DIM],
            "units": "unitless",
        },
        "cmax": {
            "name": "cmaxgrid",
            "dims": [K_DIM],
            "units": "unitless",
        },
    }

    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self._base.in_vars["data_vars"] = {
            "cx_R4": grid.x3d_compute_domain_y_dict(),
            "cy_R4": grid.y3d_compute_domain_x_dict(),
            "cmax": {},
        }
        self._base.out_vars = {
            "cmax": {},
        }
        self._stencil_factory = stencil_factory
        self._grid_data = grid
        self._quantity_factory = QuantityFactory(
            grid.sizer,
            backend=stencil_factory.backend,
        )

    def compute_parallel(self, inputs, communicator):
        self._base.make_storage_data_input_vars(inputs)
        tracer_cmax = TracerCMax(
            stencil_factory=self._stencil_factory,
            quantity_factory=self._quantity_factory,
            grid_data=self._grid_data,
            comm=communicator,
        )
        cx_quantity = self._quantity_factory.from_array(
            inputs["cx_R4"], self.inputs["cx_R4"]["dims"], ""
        )
        cy_quantity = self._quantity_factory.from_array(
            inputs["cy_R4"],
            self.inputs["cy_R4"]["dims"],
            "",
        )
        cmax_quantity = self._quantity_factory.from_array(
            inputs["cmax"],
            self.inputs["cmax"]["dims"],
            "",
        )
        tracer_cmax(
            cx=cx_quantity,
            cy=cy_quantity,
            cmax=cmax_quantity,
        )
        inputs["cmax"] = cmax_quantity[:]
        return self._base.slice_output(inputs)
