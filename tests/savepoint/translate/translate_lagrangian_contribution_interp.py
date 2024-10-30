from ndsl import StencilFactory, Namelist
from ndsl.dsl.typing import FloatField, BoolFieldIJ, IntField, IntFieldIJ, Int, Bool
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from pyFV3.stencils.map_single import lagrangian_contributions_interp

class test_Lagragian_Contribution_Interp:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid: Grid,
    ):
        print("In test_Lagragian_Contribution_interp")

        grid_indexing = stencil_factory.grid_indexing

        self._lagrangian_contributions_interp = stencil_factory.from_origin_domain(
            func=lagrangian_contributions_interp,
            origin=grid_indexing.origin_compute(),
            domain=(grid.nic, 1, grid.npz)
        )

    def __call__(
        self,
        km: int,
        not_exit_loop: BoolFieldIJ,
        INDEX_LM1: IntField,
        INDEX_LP0: IntField,
        q: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        q4_1: FloatField,
        q4_2: FloatField,
        q4_3: FloatField,
        q4_4: FloatField,
        dp1: FloatField,
        lev: IntFieldIJ,
    ):
        self._lagrangian_contributions_interp(
            km,
            not_exit_loop,
            INDEX_LM1,
            INDEX_LP0,
            q,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            lev,
        )

class TranslateLagrangian_Contribution_Interp(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.compute_func = test_Lagragian_Contribution_Interp(self.stencil_factory, self.grid)  # type: ignore
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "q1": {
                "kend": grid.npz-1,
                },

            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "q4_1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
                },
            "q4_2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_3": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_4": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "dp1_":{
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
        }

        self.out_vars = {
            "q1": {
                "kend": grid.npz-1,
                },
        }

    def compute_from_storage(self, inputs):

        self._not_exit_loop = self.quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._INDEX_LM1 = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="",
            dtype=Int,
        )

        self._INDEX_LP0 = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="",
            dtype=Int,
        )

        self._lev = self.quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="",
            dtype=Int,
        )

        self.compute_func(
                    self.grid.npz,
                    self._not_exit_loop,
                    self._INDEX_LM1,
                    self._INDEX_LP0,
                    inputs["q1"],
                    inputs["pe1_"],
                    inputs["pe2_"],
                    inputs["q4_1"],
                    inputs["q4_2"],
                    inputs["q4_3"],
                    inputs["q4_4"],
                    inputs["dp1_"],
                    self._lev,
                )

        return inputs