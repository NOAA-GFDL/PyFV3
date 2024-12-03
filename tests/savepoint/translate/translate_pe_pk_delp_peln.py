from ndsl import StencilFactory, Namelist
from ndsl.stencils.testing import TranslateFortranData2Py
from pyFV3.stencils.remapping import pe_pk_delp_peln
from ndsl.stencils.testing.grid import Grid

class TranslatePE_pk_delp_peln(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist: Namelist, stencil_factory: StencilFactory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie, 
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pe_": {
                "istart": grid.is_-1,
                "iend": grid.ie+1,
                "jstart": grid.js-1,
                "jend": grid.je+1,
                "kend": grid.npz+1,
                },
            "peln_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pn2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pk2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "delp": {
                # "istart": grid.isd,
                # "iend": grid.ied,
                # "jstart": grid.jsd,
                # "jend": grid.jed,
                # "kend": grid.npz,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "ak":{
            },
            "bk":{
            },
        }
        self.in_vars["parameters"] = [
            "akap",
            "ptop",
        ]

        self.out_vars = {
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pe_": {
                "istart": grid.is_-1,
                "iend": grid.ie+1,
                "jstart": grid.js-1,
                "jend": grid.je+1,
                "kend": grid.npz+1,
                },
            "peln_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pn2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "pk2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
            "delp": {
                # "istart": grid.isd,
                # "iend": grid.ied,
                # "jstart": grid.jsd,
                # "jend": grid.jed,
                # "kend": grid.npz,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz+1,
                },
        }

        grid_indexing = stencil_factory.grid_indexing
        self._domain_kextra = (
            grid_indexing.domain[0],
            1,
            grid_indexing.domain[2] + 1,
        )


        self._pe_pk_delp_peln = stencil_factory.from_origin_domain(
            pe_pk_delp_peln,
            origin=grid_indexing.origin_compute(),
            domain=self._domain_kextra,
        )

    def compute_from_storage(self, inputs):

        self._pe_pk_delp_peln(inputs["pe_"],
                              inputs["pk"],
                              inputs["delp"],
                              inputs["peln_"],
                              inputs["pe2_"],
                              inputs["pk2_"],
                              inputs["pn2_"],
                              inputs["ak"],
                              inputs["bk"],
                              inputs["akap"],
                              inputs["ptop"],
        )
        return inputs
