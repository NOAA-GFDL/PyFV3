from f90nml import Namelist

from ndsl import StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from pyfv3.stencils.mapn_tracer import MapNTracer
from pyfv3.tracers import FVTracersAxisName, GEOS_tracers_mapping, setup_fvtracers


class TranslateMapN_Tracer_2d(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "qtracers": {},
            "pe1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "pe2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "dp2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
        }

        self.out_vars = {
            "qtracers": {},
        }

        # Value from GEOS
        self.kord = 9

        # mode / iv set to 1 from GEOS
        self.mode = 1

        self.nq = 9

        self.fill = True

        self._tracers = None

    def compute_from_storage(self, inputs):
        setup_fvtracers(
            self.quantity_factory, inputs["qtracers"].shape[3], GEOS_tracers_mapping
        )
        self._tracers = self.quantity_factory.from_array(
            inputs["qtracers"], [I_DIM, J_DIM, K_DIM, FVTracersAxisName], ""
        )

        self._compute_func = MapNTracer(
            self.stencil_factory,
            self.quantity_factory,
            abs(self.kord),
            fill=self.fill,
        )

        self._compute_func(
            inputs["pe1"],
            inputs["pe2"],
            inputs["dp2"],
            self._tracers,
        )

        return inputs
