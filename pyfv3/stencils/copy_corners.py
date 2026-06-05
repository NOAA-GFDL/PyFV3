from functools import singledispatch

import dace
import numpy as np

from ndsl import NDSLRuntime, Quantity, StencilFactory, orchestrate
from ndsl.dsl.typing import FloatField
from ndsl.optional_imports import cupy as cp
from ndsl.dsl.dace.orchestration import dace_inhibitor


@singledispatch
def corner_copy_x(output_field, input_field):
    raise NotImplementedError(f"No CopyCorners for type {type(output_field)}")


if cp is not None:

    @corner_copy_x.register(cp.ndarray)
    def _corner_copy_x_cupy(output_field: cp.ndarray, input_field: cp.ndarray):
        _type_agnostic_copy_corners_x(output_field, input_field)


@corner_copy_x.register(np.ndarray)
def _corner_copy_x_numpy(output_field: np.ndarray, input_field: np.ndarray):
    _type_agnostic_copy_corners_x(output_field, input_field)


@corner_copy_x.register(Quantity)
def _corner_copy_x_quantity(output_field: Quantity, input_field: Quantity):
    _type_agnostic_copy_corners_x(output_field.data, input_field.data)


def _type_agnostic_copy_corners_x(output_field, input_field):
    """Equivalent to the copy_corners_x functions in fortran.

    This is written to operate on plain ndarrarys and not use the GT4Py framework.
    This choice was made because we've seen a lot of performance left on the table using
    orchestration without explicitly describing the operations but rather have full 3d-
    sweeps with conditionals.
    Since DaCe can handle (simple) operations on ndarrays directly this gives us a more
    explicit entrypoint to the language and more optimization-potential.

    Args:
        field_to_copy (ndarray): field to apply the corner copy on.
            This is explicitly not type-hinted for orchestration
    """
    output_field[0, 0] = input_field[0, 5]
    output_field[0, 1] = input_field[1, 5]
    output_field[0, 2] = input_field[2, 5]

    output_field[1, 0] = input_field[0, 4]
    output_field[1, 1] = input_field[1, 4]
    output_field[1, 2] = input_field[2, 4]

    output_field[2, 0] = input_field[0, 3]
    output_field[2, 1] = input_field[1, 3]
    output_field[2, 2] = input_field[2, 3]

    output_field[0, -4] = input_field[2, -7]
    output_field[0, -3] = input_field[1, -7]
    output_field[0, -2] = input_field[0, -7]

    output_field[1, -4] = input_field[2, -6]
    output_field[1, -3] = input_field[1, -6]
    output_field[1, -2] = input_field[0, -6]

    output_field[2, -4] = input_field[2, -5]
    output_field[2, -3] = input_field[1, -5]
    output_field[2, -2] = input_field[0, -5]

    output_field[-4, 0] = input_field[-2, 3]
    output_field[-4, 1] = input_field[-3, 3]
    output_field[-4, 2] = input_field[-4, 3]

    output_field[-3, 0] = input_field[-2, 4]
    output_field[-3, 1] = input_field[-3, 4]
    output_field[-3, 2] = input_field[-4, 4]

    output_field[-2, 0] = input_field[-2, 5]
    output_field[-2, 1] = input_field[-3, 5]
    output_field[-2, 2] = input_field[-4, 5]

    output_field[-4, -2] = input_field[-2, -5]
    output_field[-4, -3] = input_field[-3, -5]
    output_field[-4, -4] = input_field[-4, -5]

    output_field[-3, -2] = input_field[-2, -6]
    output_field[-3, -3] = input_field[-3, -6]
    output_field[-3, -4] = input_field[-4, -6]

    output_field[-2, -2] = input_field[-2, -7]
    output_field[-2, -3] = input_field[-3, -7]
    output_field[-2, -4] = input_field[-4, -7]


@singledispatch
def corner_copy_y(output_field, input_field):
    raise NotImplementedError(f"No CopyCorners for type {type(output_field)}")


if cp is not None:

    @corner_copy_y.register(cp.ndarray)
    def _corner_copy_y_cupy(output_field: cp.ndarray, input_field: cp.ndarray):
        _type_agnostic_copy_corners_y(output_field, input_field)


@corner_copy_y.register(np.ndarray)
def _corner_copy_y_numpy(output_field: np.ndarray, input_field: np.ndarray):
    _type_agnostic_copy_corners_y(output_field, input_field)


@corner_copy_y.register(Quantity)
def _corner_copy_y_quantity(output_field: Quantity, input_field: Quantity):
    _type_agnostic_copy_corners_y(output_field.data, input_field.data)


def _type_agnostic_copy_corners_y(output_field, input_field):
    """Equivalent to the copy_corners_y functions in fortran.

    This is written to operate on plain ndarrarys and not use the GT4Py framework.
    This choice was made because we've seen a lot of performance left on the table using
    orchestration without explicitly describing the operations but rather have full 3d-
    sweeps with conditionals.
    Since DaCe can handle (simple) operations on ndarrays directly this gives us a more
    explicit entrypoint to the language and more optimization-potential.

    Args:
        field_to_copy (ndarray): field to apply the corner copy on.
            This is explicitly not type-hinted for orchestration
    """
    output_field[0, 0] = input_field[5, 0]
    output_field[1, 0] = input_field[5, 1]
    output_field[2, 0] = input_field[5, 2]

    output_field[0, 1] = input_field[4, 0]
    output_field[1, 1] = input_field[4, 1]
    output_field[2, 1] = input_field[4, 2]

    output_field[0, 2] = input_field[3, 0]
    output_field[1, 2] = input_field[3, 1]
    output_field[2, 2] = input_field[3, 2]

    output_field[-4, 0] = input_field[-7, 2]
    output_field[-3, 0] = input_field[-7, 1]
    output_field[-2, 0] = input_field[-7, 0]

    output_field[-4, 1] = input_field[-6, 2]
    output_field[-3, 1] = input_field[-6, 1]
    output_field[-2, 1] = input_field[-6, 0]

    output_field[-4, 2] = input_field[-5, 2]
    output_field[-3, 2] = input_field[-5, 1]
    output_field[-2, 2] = input_field[-5, 0]

    output_field[0, -2] = input_field[5, -2]
    output_field[0, -3] = input_field[4, -2]
    output_field[0, -4] = input_field[3, -2]

    output_field[1, -2] = input_field[5, -3]
    output_field[1, -3] = input_field[4, -3]
    output_field[1, -4] = input_field[3, -3]

    output_field[2, -2] = input_field[5, -4]
    output_field[2, -3] = input_field[4, -4]
    output_field[2, -4] = input_field[3, -4]

    output_field[-2, -4] = input_field[-5, -2]
    output_field[-2, -3] = input_field[-6, -2]
    output_field[-2, -2] = input_field[-7, -2]

    output_field[-3, -4] = input_field[-5, -3]
    output_field[-3, -3] = input_field[-6, -3]
    output_field[-3, -2] = input_field[-7, -3]

    output_field[-4, -4] = input_field[-5, -4]
    output_field[-4, -3] = input_field[-6, -4]
    output_field[-4, -2] = input_field[-7, -4]


class REALCopyCornersX(NDSLRuntime):
    """
    Helper-class to copy corners corresponding to the fortran function copy_corners_x
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "Corner-Copy only implemented for exactly 3 Halo-Points"
            )

        self._is_orch = stencil_factory.backend.is_orchestrated()

    def _internal_corners_copy_per_level(self, field: FloatField, k: int):
        if self._is_orch:
            _type_agnostic_copy_corners_x(field[:, :, k], field[:, :, k])
        else:
            corner_copy_x(field, field)

    def __call__(self, field: FloatField):
        if self._is_orch:
            _type_agnostic_copy_corners_x(field, field)
        else:
            corner_copy_x(field, field)

    def nord(self, field: FloatField, nord: Quantity):
        for k in dace.map[0 : nord.shape[0]]:
            if nord[k] > 0:
                self._internal_corners_copy_per_level(field, k)


class REALCopyCornersY(NDSLRuntime):
    """
    Helper-class to copy corners corresponding to the fortran function
    copy_corners_y
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "Corner-Copy only implemented for exactly 3 Halo-Points"
            )

        self._is_orch = stencil_factory.backend.is_orchestrated()

    def _internal_corners_copy_per_level(self, field: FloatField, k: int):
        if self._is_orch:
            _type_agnostic_copy_corners_y(field[:, :, k], field[:, :, k])
        else:
            corner_copy_y(field[:, :, k], field[:, :, k])

    def __call__(self, field: FloatField):
        if self._is_orch:
            _type_agnostic_copy_corners_y(field, field)
        else:
            corner_copy_y(field, field)

    def nord(self, field: FloatField, nord: Quantity):
        for k in dace.map[0 : nord.shape[0]]:
            if nord[k] > 0:
                self._internal_corners_copy_per_level(field, k)


class CopyCornersX(NDSLRuntime):
    """
    Helper-class to copy corners corresponding to the fortran function copy_corners_x
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="nord",
        )

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "Corner-Copy only implemented for exactly 3 Halo-Points"
            )

        self._is_orch = stencil_factory.backend.is_orchestrated()

    def __call__(self, field: FloatField):
        for k in dace.map[0 : field.shape[2]]:
            field[0, 0, k] = field[0, 5, k]
            field[0, 1, k] = field[1, 5, k]
            field[0, 2, k] = field[2, 5, k]

            field[1, 0, k] = field[0, 4, k]
            field[1, 1, k] = field[1, 4, k]
            field[1, 2, k] = field[2, 4, k]

            field[2, 0, k] = field[0, 3, k]
            field[2, 1, k] = field[1, 3, k]
            field[2, 2, k] = field[2, 3, k]

            field[0, -4, k] = field[2, -7, k]
            field[0, -3, k] = field[1, -7, k]
            field[0, -2, k] = field[0, -7, k]

            field[1, -4, k] = field[2, -6, k]
            field[1, -3, k] = field[1, -6, k]
            field[1, -2, k] = field[0, -6, k]

            field[2, -4, k] = field[2, -5, k]
            field[2, -3, k] = field[1, -5, k]
            field[2, -2, k] = field[0, -5, k]

            field[-4, 0, k] = field[-2, 3, k]
            field[-4, 1, k] = field[-3, 3, k]
            field[-4, 2, k] = field[-4, 3, k]

            field[-3, 0, k] = field[-2, 4, k]
            field[-3, 1, k] = field[-3, 4, k]
            field[-3, 2, k] = field[-4, 4, k]

            field[-2, 0, k] = field[-2, 5, k]
            field[-2, 1, k] = field[-3, 5, k]
            field[-2, 2, k] = field[-4, 5, k]

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
        for k in dace.map[0 : nord.shape[0]]:
            if nord[k] > 0:
                field[0, 0, k] = field[0, 5, k]
                field[0, 1, k] = field[1, 5, k]
                field[0, 2, k] = field[2, 5, k]

                field[1, 0, k] = field[0, 4, k]
                field[1, 1, k] = field[1, 4, k]
                field[1, 2, k] = field[2, 4, k]

                field[2, 0, k] = field[0, 3, k]
                field[2, 1, k] = field[1, 3, k]
                field[2, 2, k] = field[2, 3, k]

                field[0, -4, k] = field[2, -7, k]
                field[0, -3, k] = field[1, -7, k]
                field[0, -2, k] = field[0, -7, k]

                field[1, -4, k] = field[2, -6, k]
                field[1, -3, k] = field[1, -6, k]
                field[1, -2, k] = field[0, -6, k]

                field[2, -4, k] = field[2, -5, k]
                field[2, -3, k] = field[1, -5, k]
                field[2, -2, k] = field[0, -5, k]

                field[-4, 0, k] = field[-2, 3, k]
                field[-4, 1, k] = field[-3, 3, k]
                field[-4, 2, k] = field[-4, 3, k]

                field[-3, 0, k] = field[-2, 4, k]
                field[-3, 1, k] = field[-3, 4, k]
                field[-3, 2, k] = field[-4, 4, k]

                field[-2, 0, k] = field[-2, 5, k]
                field[-2, 1, k] = field[-3, 5, k]
                field[-2, 2, k] = field[-4, 5, k]

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
    Helper-class to copy corners corresponding to the fortran function
    copy_corners_y
    """

    def __init__(self, stencil_factory: StencilFactory) -> None:
        super().__init__(stencil_factory)
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
            method_to_orchestrate="nord",
        )

        if stencil_factory.grid_indexing.n_halo != 3:
            raise NotImplementedError(
                "Corner-Copy only implemented for exactly 3 Halo-Points"
            )

        self._is_orch = stencil_factory.backend.is_orchestrated()

    def __call__(self, field: FloatField):
        for k in dace.map[0 : field.shape[2]]:
            field[0, 0, k] = field[5, 0, k]
            field[1, 0, k] = field[5, 1, k]
            field[2, 0, k] = field[5, 2, k]

            field[0, 1, k] = field[4, 0, k]
            field[1, 1, k] = field[4, 1, k]
            field[2, 1, k] = field[4, 2, k]

            field[0, 2, k] = field[3, 0, k]
            field[1, 2, k] = field[3, 1, k]
            field[2, 2, k] = field[3, 2, k]

            field[-4, 0, k] = field[-7, 2, k]
            field[-3, 0, k] = field[-7, 1, k]
            field[-2, 0, k] = field[-7, 0, k]

            field[-4, 1, k] = field[-6, 2, k]
            field[-3, 1, k] = field[-6, 1, k]
            field[-2, 1, k] = field[-6, 0, k]

            field[-4, 2, k] = field[-5, 2, k]
            field[-3, 2, k] = field[-5, 1, k]
            field[-2, 2, k] = field[-5, 0, k]

            field[0, -2, k] = field[5, -2, k]
            field[0, -3, k] = field[4, -2, k]
            field[0, -4, k] = field[3, -2, k]

            field[1, -2, k] = field[5, -3, k]
            field[1, -3, k] = field[4, -3, k]
            field[1, -4, k] = field[3, -3, k]

            field[2, -2, k] = field[5, -4, k]
            field[2, -3, k] = field[4, -4, k]
            field[2, -4, k] = field[3, -4, k]

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
        for k in dace.map[0 : nord.shape[0]]:
            if nord[k] > 0:
                field[0, 0, k] = field[5, 0, k]
                field[1, 0, k] = field[5, 1, k]
                field[2, 0, k] = field[5, 2, k]

                field[0, 1, k] = field[4, 0, k]
                field[1, 1, k] = field[4, 1, k]
                field[2, 1, k] = field[4, 2, k]

                field[0, 2, k] = field[3, 0, k]
                field[1, 2, k] = field[3, 1, k]
                field[2, 2, k] = field[3, 2, k]

                field[-4, 0, k] = field[-7, 2, k]
                field[-3, 0, k] = field[-7, 1, k]
                field[-2, 0, k] = field[-7, 0, k]

                field[-4, 1, k] = field[-6, 2, k]
                field[-3, 1, k] = field[-6, 1, k]
                field[-2, 1, k] = field[-6, 0, k]

                field[-4, 2, k] = field[-5, 2, k]
                field[-3, 2, k] = field[-5, 1, k]
                field[-2, 2, k] = field[-5, 0, k]

                field[0, -2, k] = field[5, -2, k]
                field[0, -3, k] = field[4, -2, k]
                field[0, -4, k] = field[3, -2, k]

                field[1, -2, k] = field[5, -3, k]
                field[1, -3, k] = field[4, -3, k]
                field[1, -4, k] = field[3, -3, k]

                field[2, -2, k] = field[5, -4, k]
                field[2, -3, k] = field[4, -4, k]
                field[2, -4, k] = field[3, -4, k]

                field[-2, -4, k] = field[-5, -2, k]
                field[-2, -3, k] = field[-6, -2, k]
                field[-2, -2, k] = field[-7, -2, k]

                field[-3, -4, k] = field[-5, -3, k]
                field[-3, -3, k] = field[-6, -3, k]
                field[-3, -2, k] = field[-7, -3, k]

                field[-4, -4, k] = field[-5, -4, k]
                field[-4, -3, k] = field[-6, -4, k]
                field[-4, -2, k] = field[-7, -4, k]
