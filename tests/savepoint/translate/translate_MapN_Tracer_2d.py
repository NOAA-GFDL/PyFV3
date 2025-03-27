from ndsl import Namelist, StencilFactory
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from pyFV3.stencils.mapn_tracer import MapNTracer


class TranslateMapN_Tracer_2d(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
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
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
            },
            "dp2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz - 1,
            },
        }

        self.out_vars = {
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
        }

        # Value from GEOS
        self.kord = 9

        # mode / iv set to 1 from GEOS
        self.mode = 1

        self.nq = 9

        self.fill = True

    def compute_from_storage(self, inputs):

        tracers = {
            "qvapor": inputs["qvapor"],
            "qliquid": inputs["qliquid"],
            "qice": inputs["qice"],
            "qrain": inputs["qrain"],
            "qsnow": inputs["qsnow"],
            "qgraupel": inputs["qgraupel"],
            "qcld": inputs["qcld"],
            "qo3mr": inputs["qo3mr"],
            "qsgs_tke": inputs["qsgs_tke"],
        }

        self._compute_func = MapNTracer(
            self.stencil_factory,
            self.quantity_factory,
            abs(self.kord),
            self.nq,
            fill=self.fill,
            tracers=tracers,
        )

        self._compute_func(
            inputs["pe1_"],
            inputs["pe2_"],
            inputs["dp2_"],
            tracers,
        )

        return inputs
