import dace

from ndsl import NDSLRuntime, Quantity, StencilFactory, orchestrate
from ndsl.dsl.typing import FloatField


class CopyCornersX(NDSLRuntime):
    """
    Helper-class to copy corners corresponding to the fortran function `copy_corners_x`.
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="nord",
        )
        self._is_ne_corner = stencil_factory.grid_indexing.ne_corner
        self._is_nw_corner = stencil_factory.grid_indexing.nw_corner
        self._is_sw_corner = stencil_factory.grid_indexing.sw_corner
        self._is_se_corner = stencil_factory.grid_indexing.se_corner

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "CopyCornersX is only implemented for exactly 3 halo points."
            )

    def __call__(self, field: FloatField):
        for __k in dace.map[0 : field.shape[2]]:
            if self._is_sw_corner:
                field[0, 0, __k] = field[0, 5, __k]
                field[0, 1, __k] = field[1, 5, __k]
                field[0, 2, __k] = field[2, 5, __k]

                field[1, 0, __k] = field[0, 4, __k]
                field[1, 1, __k] = field[1, 4, __k]
                field[1, 2, __k] = field[2, 4, __k]

                field[2, 0, __k] = field[0, 3, __k]
                field[2, 1, __k] = field[1, 3, __k]
                field[2, 2, __k] = field[2, 3, __k]

            if self._is_se_corner:
                field[-4, 0, __k] = field[-2, 3, __k]
                field[-4, 1, __k] = field[-3, 3, __k]
                field[-4, 2, __k] = field[-4, 3, __k]

                field[-3, 0, __k] = field[-2, 4, __k]
                field[-3, 1, __k] = field[-3, 4, __k]
                field[-3, 2, __k] = field[-4, 4, __k]

                field[-2, 0, __k] = field[-2, 5, __k]
                field[-2, 1, __k] = field[-3, 5, __k]
                field[-2, 2, __k] = field[-4, 5, __k]

            if self._is_nw_corner:
                field[0, -4, __k] = field[2, -7, __k]
                field[0, -3, __k] = field[1, -7, __k]
                field[0, -2, __k] = field[0, -7, __k]

                field[1, -4, __k] = field[2, -6, __k]
                field[1, -3, __k] = field[1, -6, __k]
                field[1, -2, __k] = field[0, -6, __k]

                field[2, -4, __k] = field[2, -5, __k]
                field[2, -3, __k] = field[1, -5, __k]
                field[2, -2, __k] = field[0, -5, __k]

            if self._is_ne_corner:
                field[-4, -2, __k] = field[-2, -5, __k]
                field[-4, -3, __k] = field[-3, -5, __k]
                field[-4, -4, __k] = field[-4, -5, __k]

                field[-3, -2, __k] = field[-2, -6, __k]
                field[-3, -3, __k] = field[-3, -6, __k]
                field[-3, -4, __k] = field[-4, -6, __k]

                field[-2, -2, __k] = field[-2, -7, __k]
                field[-2, -3, __k] = field[-3, -7, __k]
                field[-2, -4, __k] = field[-4, -7, __k]

    def nord(self, field: FloatField, nord: Quantity):
        for __k in dace.map[0 : nord.shape[0]]:
            if nord[__k] > 0:
                if self._is_sw_corner:
                    field[0, 0, __k] = field[0, 5, __k]
                    field[0, 1, __k] = field[1, 5, __k]
                    field[0, 2, __k] = field[2, 5, __k]

                    field[1, 0, __k] = field[0, 4, __k]
                    field[1, 1, __k] = field[1, 4, __k]
                    field[1, 2, __k] = field[2, 4, __k]

                    field[2, 0, __k] = field[0, 3, __k]
                    field[2, 1, __k] = field[1, 3, __k]
                    field[2, 2, __k] = field[2, 3, __k]

                if self._is_se_corner:
                    field[-4, 0, __k] = field[-2, 3, __k]
                    field[-4, 1, __k] = field[-3, 3, __k]
                    field[-4, 2, __k] = field[-4, 3, __k]

                    field[-3, 0, __k] = field[-2, 4, __k]
                    field[-3, 1, __k] = field[-3, 4, __k]
                    field[-3, 2, __k] = field[-4, 4, __k]

                    field[-2, 0, __k] = field[-2, 5, __k]
                    field[-2, 1, __k] = field[-3, 5, __k]
                    field[-2, 2, __k] = field[-4, 5, __k]

                if self._is_nw_corner:
                    field[0, -4, __k] = field[2, -7, __k]
                    field[0, -3, __k] = field[1, -7, __k]
                    field[0, -2, __k] = field[0, -7, __k]

                    field[1, -4, __k] = field[2, -6, __k]
                    field[1, -3, __k] = field[1, -6, __k]
                    field[1, -2, __k] = field[0, -6, __k]

                    field[2, -4, __k] = field[2, -5, __k]
                    field[2, -3, __k] = field[1, -5, __k]
                    field[2, -2, __k] = field[0, -5, __k]

                if self._is_ne_corner:
                    field[-4, -2, __k] = field[-2, -5, __k]
                    field[-4, -3, __k] = field[-3, -5, __k]
                    field[-4, -4, __k] = field[-4, -5, __k]

                    field[-3, -2, __k] = field[-2, -6, __k]
                    field[-3, -3, __k] = field[-3, -6, __k]
                    field[-3, -4, __k] = field[-4, -6, __k]

                    field[-2, -2, __k] = field[-2, -7, __k]
                    field[-2, -3, __k] = field[-3, -7, __k]
                    field[-2, -4, __k] = field[-4, -7, __k]


class CopyCornersY(NDSLRuntime):
    """
    Helper-class to copy corners corresponding to the Fortran function `copy_corners_y`.
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="nord",
        )

        self._is_ne_corner = stencil_factory.grid_indexing.ne_corner
        self._is_nw_corner = stencil_factory.grid_indexing.nw_corner
        self._is_sw_corner = stencil_factory.grid_indexing.sw_corner
        self._is_se_corner = stencil_factory.grid_indexing.se_corner

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "CopyCornersY is only implemented for exactly 3 halo points."
            )

    def __call__(self, field: FloatField):
        for __k in dace.map[0 : field.shape[2]]:
            if self._is_sw_corner:
                field[0, 0, __k] = field[5, 0, __k]
                field[1, 0, __k] = field[5, 1, __k]
                field[2, 0, __k] = field[5, 2, __k]

                field[0, 1, __k] = field[4, 0, __k]
                field[1, 1, __k] = field[4, 1, __k]
                field[2, 1, __k] = field[4, 2, __k]

                field[0, 2, __k] = field[3, 0, __k]
                field[1, 2, __k] = field[3, 1, __k]
                field[2, 2, __k] = field[3, 2, __k]

            if self._is_se_corner:
                field[-4, 0, __k] = field[-7, 2, __k]
                field[-3, 0, __k] = field[-7, 1, __k]
                field[-2, 0, __k] = field[-7, 0, __k]

                field[-4, 1, __k] = field[-6, 2, __k]
                field[-3, 1, __k] = field[-6, 1, __k]
                field[-2, 1, __k] = field[-6, 0, __k]

                field[-4, 2, __k] = field[-5, 2, __k]
                field[-3, 2, __k] = field[-5, 1, __k]
                field[-2, 2, __k] = field[-5, 0, __k]

            if self._is_nw_corner:
                field[0, -2, __k] = field[5, -2, __k]
                field[0, -3, __k] = field[4, -2, __k]
                field[0, -4, __k] = field[3, -2, __k]

                field[1, -2, __k] = field[5, -3, __k]
                field[1, -3, __k] = field[4, -3, __k]
                field[1, -4, __k] = field[3, -3, __k]

                field[2, -2, __k] = field[5, -4, __k]
                field[2, -3, __k] = field[4, -4, __k]
                field[2, -4, __k] = field[3, -4, __k]

            if self._is_ne_corner:
                field[-2, -4, __k] = field[-5, -2, __k]
                field[-2, -3, __k] = field[-6, -2, __k]
                field[-2, -2, __k] = field[-7, -2, __k]

                field[-3, -4, __k] = field[-5, -3, __k]
                field[-3, -3, __k] = field[-6, -3, __k]
                field[-3, -2, __k] = field[-7, -3, __k]

                field[-4, -4, __k] = field[-5, -4, __k]
                field[-4, -3, __k] = field[-6, -4, __k]
                field[-4, -2, __k] = field[-7, -4, __k]

    def nord(self, field: FloatField, nord: Quantity):
        for __k in dace.map[0 : nord.shape[0]]:
            if nord[__k] > 0:
                if self._is_sw_corner:
                    field[0, 0, __k] = field[5, 0, __k]
                    field[1, 0, __k] = field[5, 1, __k]
                    field[2, 0, __k] = field[5, 2, __k]

                    field[0, 1, __k] = field[4, 0, __k]
                    field[1, 1, __k] = field[4, 1, __k]
                    field[2, 1, __k] = field[4, 2, __k]

                    field[0, 2, __k] = field[3, 0, __k]
                    field[1, 2, __k] = field[3, 1, __k]
                    field[2, 2, __k] = field[3, 2, __k]

                if self._is_se_corner:
                    field[-4, 0, __k] = field[-7, 2, __k]
                    field[-3, 0, __k] = field[-7, 1, __k]
                    field[-2, 0, __k] = field[-7, 0, __k]

                    field[-4, 1, __k] = field[-6, 2, __k]
                    field[-3, 1, __k] = field[-6, 1, __k]
                    field[-2, 1, __k] = field[-6, 0, __k]

                    field[-4, 2, __k] = field[-5, 2, __k]
                    field[-3, 2, __k] = field[-5, 1, __k]
                    field[-2, 2, __k] = field[-5, 0, __k]

                if self._is_nw_corner:
                    field[0, -2, __k] = field[5, -2, __k]
                    field[0, -3, __k] = field[4, -2, __k]
                    field[0, -4, __k] = field[3, -2, __k]

                    field[1, -2, __k] = field[5, -3, __k]
                    field[1, -3, __k] = field[4, -3, __k]
                    field[1, -4, __k] = field[3, -3, __k]

                    field[2, -2, __k] = field[5, -4, __k]
                    field[2, -3, __k] = field[4, -4, __k]
                    field[2, -4, __k] = field[3, -4, __k]

                if self._is_ne_corner:
                    field[-2, -4, __k] = field[-5, -2, __k]
                    field[-2, -3, __k] = field[-6, -2, __k]
                    field[-2, -2, __k] = field[-7, -2, __k]

                    field[-3, -4, __k] = field[-5, -3, __k]
                    field[-3, -3, __k] = field[-6, -3, __k]
                    field[-3, -2, __k] = field[-7, -3, __k]

                    field[-4, -4, __k] = field[-5, -4, __k]
                    field[-4, -3, __k] = field[-6, -4, __k]
                    field[-4, -2, __k] = field[-7, -4, __k]
