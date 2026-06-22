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
        self._on_gpu = stencil_factory.backend.is_gpu_backend()

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "CopyCornersX is only implemented for exactly 3 halo points."
            )

    def __call__(self, field: FloatField):
        if not self._on_gpu:
            for k in dace.map[0 : field.shape[2]]:
                if self._is_sw_corner:
                    field[0, 0, k] = field[0, 5, k]
                    field[0, 1, k] = field[1, 5, k]
                    field[0, 2, k] = field[2, 5, k]

                    field[1, 0, k] = field[0, 4, k]
                    field[1, 1, k] = field[1, 4, k]
                    field[1, 2, k] = field[2, 4, k]

                    field[2, 0, k] = field[0, 3, k]
                    field[2, 1, k] = field[1, 3, k]
                    field[2, 2, k] = field[2, 3, k]

                if self._is_se_corner:
                    field[-4, 0, k] = field[-2, 3, k]
                    field[-4, 1, k] = field[-3, 3, k]
                    field[-4, 2, k] = field[-4, 3, k]

                    field[-3, 0, k] = field[-2, 4, k]
                    field[-3, 1, k] = field[-3, 4, k]
                    field[-3, 2, k] = field[-4, 4, k]

                    field[-2, 0, k] = field[-2, 5, k]
                    field[-2, 1, k] = field[-3, 5, k]
                    field[-2, 2, k] = field[-4, 5, k]

                if self._is_nw_corner:
                    field[0, -4, k] = field[2, -7, k]
                    field[0, -3, k] = field[1, -7, k]
                    field[0, -2, k] = field[0, -7, k]

                    field[1, -4, k] = field[2, -6, k]
                    field[1, -3, k] = field[1, -6, k]
                    field[1, -2, k] = field[0, -6, k]

                    field[2, -4, k] = field[2, -5, k]
                    field[2, -3, k] = field[1, -5, k]
                    field[2, -2, k] = field[0, -5, k]

                if self._is_ne_corner:
                    field[-4, -2, k] = field[-2, -5, k]
                    field[-4, -3, k] = field[-3, -5, k]
                    field[-4, -4, k] = field[-4, -5, k]

                    field[-3, -2, k] = field[-2, -6, k]
                    field[-3, -3, k] = field[-3, -6, k]
                    field[-3, -4, k] = field[-4, -6, k]

                    field[-2, -2, k] = field[-2, -7, k]
                    field[-2, -3, k] = field[-3, -7, k]
                    field[-2, -4, k] = field[-4, -7, k]
        else:
            for k in dace.map[0 : field.shape[2]] @ dace.ScheduleType.GPU_Device:
                if self._is_sw_corner:
                    field[0, 0, k] = field[0, 5, k]
                    field[0, 1, k] = field[1, 5, k]
                    field[0, 2, k] = field[2, 5, k]

                    field[1, 0, k] = field[0, 4, k]
                    field[1, 1, k] = field[1, 4, k]
                    field[1, 2, k] = field[2, 4, k]

                    field[2, 0, k] = field[0, 3, k]
                    field[2, 1, k] = field[1, 3, k]
                    field[2, 2, k] = field[2, 3, k]

                if self._is_se_corner:
                    field[-4, 0, k] = field[-2, 3, k]
                    field[-4, 1, k] = field[-3, 3, k]
                    field[-4, 2, k] = field[-4, 3, k]

                    field[-3, 0, k] = field[-2, 4, k]
                    field[-3, 1, k] = field[-3, 4, k]
                    field[-3, 2, k] = field[-4, 4, k]

                    field[-2, 0, k] = field[-2, 5, k]
                    field[-2, 1, k] = field[-3, 5, k]
                    field[-2, 2, k] = field[-4, 5, k]

                if self._is_nw_corner:
                    field[0, -4, k] = field[2, -7, k]
                    field[0, -3, k] = field[1, -7, k]
                    field[0, -2, k] = field[0, -7, k]

                    field[1, -4, k] = field[2, -6, k]
                    field[1, -3, k] = field[1, -6, k]
                    field[1, -2, k] = field[0, -6, k]

                    field[2, -4, k] = field[2, -5, k]
                    field[2, -3, k] = field[1, -5, k]
                    field[2, -2, k] = field[0, -5, k]

                if self._is_ne_corner:
                    field[-4, -2, k] = field[-2, -5, k]
                    field[-4, -3, k] = field[-3, -5, k]
                    field[-4, -4, k] = field[-4, -5, k]

                    field[-3, -2, k] = field[-2, -6, k]
                    field[-3, -3, k] = field[-3, -6, k]
                    field[-3, -4, k] = field[-4, -6, k]

                    field[-2, -2, k] = field[-2, -7, k]
                    field[-2, -3, k] = field[-3, -7, k]
                    field[-2, -4, k] = field[-4, -7, k]

    def nord(self, field: FloatField, nord: Quantity):
        if not self._on_gpu:
            for k in dace.map[0 : nord.shape[0]]:
                if nord[k] > 0:
                    if self._is_sw_corner:
                        field[0, 0, k] = field[0, 5, k]
                        field[0, 1, k] = field[1, 5, k]
                        field[0, 2, k] = field[2, 5, k]

                        field[1, 0, k] = field[0, 4, k]
                        field[1, 1, k] = field[1, 4, k]
                        field[1, 2, k] = field[2, 4, k]

                        field[2, 0, k] = field[0, 3, k]
                        field[2, 1, k] = field[1, 3, k]
                        field[2, 2, k] = field[2, 3, k]

                    if self._is_se_corner:
                        field[-4, 0, k] = field[-2, 3, k]
                        field[-4, 1, k] = field[-3, 3, k]
                        field[-4, 2, k] = field[-4, 3, k]

                        field[-3, 0, k] = field[-2, 4, k]
                        field[-3, 1, k] = field[-3, 4, k]
                        field[-3, 2, k] = field[-4, 4, k]

                        field[-2, 0, k] = field[-2, 5, k]
                        field[-2, 1, k] = field[-3, 5, k]
                        field[-2, 2, k] = field[-4, 5, k]

                    if self._is_nw_corner:
                        field[0, -4, k] = field[2, -7, k]
                        field[0, -3, k] = field[1, -7, k]
                        field[0, -2, k] = field[0, -7, k]

                        field[1, -4, k] = field[2, -6, k]
                        field[1, -3, k] = field[1, -6, k]
                        field[1, -2, k] = field[0, -6, k]

                        field[2, -4, k] = field[2, -5, k]
                        field[2, -3, k] = field[1, -5, k]
                        field[2, -2, k] = field[0, -5, k]

                    if self._is_ne_corner:
                        field[-4, -2, k] = field[-2, -5, k]
                        field[-4, -3, k] = field[-3, -5, k]
                        field[-4, -4, k] = field[-4, -5, k]

                        field[-3, -2, k] = field[-2, -6, k]
                        field[-3, -3, k] = field[-3, -6, k]
                        field[-3, -4, k] = field[-4, -6, k]

                        field[-2, -2, k] = field[-2, -7, k]
                        field[-2, -3, k] = field[-3, -7, k]
                        field[-2, -4, k] = field[-4, -7, k]
        else:
            for k in dace.map[0 : nord.shape[0]] @ dace.ScheduleType.GPU_Device:
                if nord[k] > 0:
                    if self._is_sw_corner:
                        field[0, 0, k] = field[0, 5, k]
                        field[0, 1, k] = field[1, 5, k]
                        field[0, 2, k] = field[2, 5, k]

                        field[1, 0, k] = field[0, 4, k]
                        field[1, 1, k] = field[1, 4, k]
                        field[1, 2, k] = field[2, 4, k]

                        field[2, 0, k] = field[0, 3, k]
                        field[2, 1, k] = field[1, 3, k]
                        field[2, 2, k] = field[2, 3, k]

                    if self._is_se_corner:
                        field[-4, 0, k] = field[-2, 3, k]
                        field[-4, 1, k] = field[-3, 3, k]
                        field[-4, 2, k] = field[-4, 3, k]

                        field[-3, 0, k] = field[-2, 4, k]
                        field[-3, 1, k] = field[-3, 4, k]
                        field[-3, 2, k] = field[-4, 4, k]

                        field[-2, 0, k] = field[-2, 5, k]
                        field[-2, 1, k] = field[-3, 5, k]
                        field[-2, 2, k] = field[-4, 5, k]

                    if self._is_nw_corner:
                        field[0, -4, k] = field[2, -7, k]
                        field[0, -3, k] = field[1, -7, k]
                        field[0, -2, k] = field[0, -7, k]

                        field[1, -4, k] = field[2, -6, k]
                        field[1, -3, k] = field[1, -6, k]
                        field[1, -2, k] = field[0, -6, k]

                        field[2, -4, k] = field[2, -5, k]
                        field[2, -3, k] = field[1, -5, k]
                        field[2, -2, k] = field[0, -5, k]

                    if self._is_ne_corner:
                        field[-4, -2, k] = field[-2, -5, k]
                        field[-4, -3, k] = field[-3, -5, k]
                        field[-4, -4, k] = field[-4, -5, k]

                        field[-3, -2, k] = field[-2, -6, k]
                        field[-3, -3, k] = field[-3, -6, k]
                        field[-3, -4, k] = field[-4, -6, k]

                        field[-2, -2, k] = field[-2, -7, k]
                        field[-2, -3, k] = field[-3, -7, k]
                        field[-2, -4, k] = field[-4, -7, k]


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
        self._on_gpu = stencil_factory.backend.is_gpu_backend()

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "CopyCornersY is only implemented for exactly 3 halo points."
            )

    def __call__(self, field: FloatField):
        if not self._on_gpu:
            for k in dace.map[0 : field.shape[2]]:
                if self._is_sw_corner:
                    field[0, 0, k] = field[5, 0, k]
                    field[1, 0, k] = field[5, 1, k]
                    field[2, 0, k] = field[5, 2, k]

                    field[0, 1, k] = field[4, 0, k]
                    field[1, 1, k] = field[4, 1, k]
                    field[2, 1, k] = field[4, 2, k]

                    field[0, 2, k] = field[3, 0, k]
                    field[1, 2, k] = field[3, 1, k]
                    field[2, 2, k] = field[3, 2, k]

                if self._is_se_corner:
                    field[-4, 0, k] = field[-7, 2, k]
                    field[-3, 0, k] = field[-7, 1, k]
                    field[-2, 0, k] = field[-7, 0, k]

                    field[-4, 1, k] = field[-6, 2, k]
                    field[-3, 1, k] = field[-6, 1, k]
                    field[-2, 1, k] = field[-6, 0, k]

                    field[-4, 2, k] = field[-5, 2, k]
                    field[-3, 2, k] = field[-5, 1, k]
                    field[-2, 2, k] = field[-5, 0, k]

                if self._is_nw_corner:
                    field[0, -2, k] = field[5, -2, k]
                    field[0, -3, k] = field[4, -2, k]
                    field[0, -4, k] = field[3, -2, k]

                    field[1, -2, k] = field[5, -3, k]
                    field[1, -3, k] = field[4, -3, k]
                    field[1, -4, k] = field[3, -3, k]

                    field[2, -2, k] = field[5, -4, k]
                    field[2, -3, k] = field[4, -4, k]
                    field[2, -4, k] = field[3, -4, k]

                if self._is_ne_corner:
                    field[-2, -4, k] = field[-5, -2, k]
                    field[-2, -3, k] = field[-6, -2, k]
                    field[-2, -2, k] = field[-7, -2, k]

                    field[-3, -4, k] = field[-5, -3, k]
                    field[-3, -3, k] = field[-6, -3, k]
                    field[-3, -2, k] = field[-7, -3, k]

                    field[-4, -4, k] = field[-5, -4, k]
                    field[-4, -3, k] = field[-6, -4, k]
                    field[-4, -2, k] = field[-7, -4, k]
        else:
            for k in dace.map[0 : field.shape[2]] @ dace.ScheduleType.GPU_Device:
                if self._is_sw_corner:
                    field[0, 0, k] = field[5, 0, k]
                    field[1, 0, k] = field[5, 1, k]
                    field[2, 0, k] = field[5, 2, k]

                    field[0, 1, k] = field[4, 0, k]
                    field[1, 1, k] = field[4, 1, k]
                    field[2, 1, k] = field[4, 2, k]

                    field[0, 2, k] = field[3, 0, k]
                    field[1, 2, k] = field[3, 1, k]
                    field[2, 2, k] = field[3, 2, k]

                if self._is_se_corner:
                    field[-4, 0, k] = field[-7, 2, k]
                    field[-3, 0, k] = field[-7, 1, k]
                    field[-2, 0, k] = field[-7, 0, k]

                    field[-4, 1, k] = field[-6, 2, k]
                    field[-3, 1, k] = field[-6, 1, k]
                    field[-2, 1, k] = field[-6, 0, k]

                    field[-4, 2, k] = field[-5, 2, k]
                    field[-3, 2, k] = field[-5, 1, k]
                    field[-2, 2, k] = field[-5, 0, k]

                if self._is_nw_corner:
                    field[0, -2, k] = field[5, -2, k]
                    field[0, -3, k] = field[4, -2, k]
                    field[0, -4, k] = field[3, -2, k]

                    field[1, -2, k] = field[5, -3, k]
                    field[1, -3, k] = field[4, -3, k]
                    field[1, -4, k] = field[3, -3, k]

                    field[2, -2, k] = field[5, -4, k]
                    field[2, -3, k] = field[4, -4, k]
                    field[2, -4, k] = field[3, -4, k]

                if self._is_ne_corner:
                    field[-2, -4, k] = field[-5, -2, k]
                    field[-2, -3, k] = field[-6, -2, k]
                    field[-2, -2, k] = field[-7, -2, k]

                    field[-3, -4, k] = field[-5, -3, k]
                    field[-3, -3, k] = field[-6, -3, k]
                    field[-3, -2, k] = field[-7, -3, k]

                    field[-4, -4, k] = field[-5, -4, k]
                    field[-4, -3, k] = field[-6, -4, k]
                    field[-4, -2, k] = field[-7, -4, k]

    def nord(self, field: FloatField, nord: Quantity):
        if not self._on_gpu:
            for k in dace.map[0 : nord.shape[0]]:
                if nord[k] > 0:
                    if self._is_sw_corner:
                        field[0, 0, k] = field[5, 0, k]
                        field[1, 0, k] = field[5, 1, k]
                        field[2, 0, k] = field[5, 2, k]

                        field[0, 1, k] = field[4, 0, k]
                        field[1, 1, k] = field[4, 1, k]
                        field[2, 1, k] = field[4, 2, k]

                        field[0, 2, k] = field[3, 0, k]
                        field[1, 2, k] = field[3, 1, k]
                        field[2, 2, k] = field[3, 2, k]

                    if self._is_se_corner:
                        field[-4, 0, k] = field[-7, 2, k]
                        field[-3, 0, k] = field[-7, 1, k]
                        field[-2, 0, k] = field[-7, 0, k]

                        field[-4, 1, k] = field[-6, 2, k]
                        field[-3, 1, k] = field[-6, 1, k]
                        field[-2, 1, k] = field[-6, 0, k]

                        field[-4, 2, k] = field[-5, 2, k]
                        field[-3, 2, k] = field[-5, 1, k]
                        field[-2, 2, k] = field[-5, 0, k]

                    if self._is_nw_corner:
                        field[0, -2, k] = field[5, -2, k]
                        field[0, -3, k] = field[4, -2, k]
                        field[0, -4, k] = field[3, -2, k]

                        field[1, -2, k] = field[5, -3, k]
                        field[1, -3, k] = field[4, -3, k]
                        field[1, -4, k] = field[3, -3, k]

                        field[2, -2, k] = field[5, -4, k]
                        field[2, -3, k] = field[4, -4, k]
                        field[2, -4, k] = field[3, -4, k]

                    if self._is_ne_corner:
                        field[-2, -4, k] = field[-5, -2, k]
                        field[-2, -3, k] = field[-6, -2, k]
                        field[-2, -2, k] = field[-7, -2, k]

                        field[-3, -4, k] = field[-5, -3, k]
                        field[-3, -3, k] = field[-6, -3, k]
                        field[-3, -2, k] = field[-7, -3, k]

                        field[-4, -4, k] = field[-5, -4, k]
                        field[-4, -3, k] = field[-6, -4, k]
                        field[-4, -2, k] = field[-7, -4, k]
        else:
            for k in dace.map[0 : nord.shape[0]] @ dace.ScheduleType.GPU_Device:
                if nord[k] > 0:
                    if self._is_sw_corner:
                        field[0, 0, k] = field[5, 0, k]
                        field[1, 0, k] = field[5, 1, k]
                        field[2, 0, k] = field[5, 2, k]

                        field[0, 1, k] = field[4, 0, k]
                        field[1, 1, k] = field[4, 1, k]
                        field[2, 1, k] = field[4, 2, k]

                        field[0, 2, k] = field[3, 0, k]
                        field[1, 2, k] = field[3, 1, k]
                        field[2, 2, k] = field[3, 2, k]

                    if self._is_se_corner:
                        field[-4, 0, k] = field[-7, 2, k]
                        field[-3, 0, k] = field[-7, 1, k]
                        field[-2, 0, k] = field[-7, 0, k]

                        field[-4, 1, k] = field[-6, 2, k]
                        field[-3, 1, k] = field[-6, 1, k]
                        field[-2, 1, k] = field[-6, 0, k]

                        field[-4, 2, k] = field[-5, 2, k]
                        field[-3, 2, k] = field[-5, 1, k]
                        field[-2, 2, k] = field[-5, 0, k]

                    if self._is_nw_corner:
                        field[0, -2, k] = field[5, -2, k]
                        field[0, -3, k] = field[4, -2, k]
                        field[0, -4, k] = field[3, -2, k]

                        field[1, -2, k] = field[5, -3, k]
                        field[1, -3, k] = field[4, -3, k]
                        field[1, -4, k] = field[3, -3, k]

                        field[2, -2, k] = field[5, -4, k]
                        field[2, -3, k] = field[4, -4, k]
                        field[2, -4, k] = field[3, -4, k]

                    if self._is_ne_corner:
                        field[-2, -4, k] = field[-5, -2, k]
                        field[-2, -3, k] = field[-6, -2, k]
                        field[-2, -2, k] = field[-7, -2, k]

                        field[-3, -4, k] = field[-5, -3, k]
                        field[-3, -3, k] = field[-6, -3, k]
                        field[-3, -2, k] = field[-7, -3, k]

                        field[-4, -4, k] = field[-5, -4, k]
                        field[-4, -3, k] = field[-6, -4, k]
                        field[-4, -2, k] = field[-7, -4, k]
