from ndsl import StencilFactory
from f90nml import Namelist
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from pyFV3.stencils.mapn_tracer import MapNTracer
from pyFV3.tracers import setup_tracers


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

        self._are_tracers_setup = False

        self._tracers = None

    def compute_from_storage(self, inputs):
        if not self._are_tracers_setup:
            self._are_tracers_setup = True
            self._tracers = setup_tracers(
                number_of_tracers=inputs["qtracers"].shape[3],
                quantity_factory=self.quantity_factory,
                # mappings={"cloud": 6},
            )
            # tracers.quantity.data[:-1, :-1, :-1, :] = inputs["qtracers"]
        self._tracers.quantity.data = inputs["qtracers"]

        self._compute_func = MapNTracer(
            self.stencil_factory,
            self.quantity_factory,
            abs(self.kord),
            fill=self.fill,
            tracers=self._tracers,
        )

        self._compute_func(
            inputs["pe1"],
            inputs["pe2"],
            inputs["dp2"],
            self._tracers,
        )

        return inputs
