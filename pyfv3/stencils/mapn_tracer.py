import dace

from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import FloatField
from pyfv3.stencils.fillz import FillNegativeTracerValues
from pyfv3.stencils.map_single import MapSingle


class MapNTracer(NDSLRuntime):
    """
    Fortran code is mapn_tracer, test class is MapN_Tracer_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        kord: int,
        fill: bool,
        tracers,
    ):
        super().__init__(stencil_factory)
        self._nq = int(tracers.shape[3])

        self._map_single_parametrized_kord = MapSingle(
            stencil_factory,
            quantity_factory,
            kord,
            0,
            dims=[I_DIM, J_DIM, K_DIM],
        )

        self._map_single_kord9 = MapSingle(
            stencil_factory,
            quantity_factory,
            9,
            0,
            dims=[I_DIM, J_DIM, K_DIM],
        )

        if fill:
            self._fill_negative_tracers = True
            self._fillz = FillNegativeTracerValues(
                stencil_factory,
                quantity_factory,
                self._nq,
            )
        else:
            self._fill_negative_tracers = False

        self._index_cloud = tracers.index("cloud")

    def __call__(
        self,
        pe1: FloatField,
        pe2: FloatField,
        dp2: FloatField,
        tracers,
    ):
        """
        Remaps the tracer species onto the Eulerian grid
        and optionally fills negative values in the tracer fields
        Assumes the minimum value is 0 for each tracer

        Args:
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            dp2 (in): Difference in pressure between Eulerian levels
            tracers (inout): tracers to be remapped
        """
        for i_tracer in dace.nounroll(range(tracers.shape[3])):
            if i_tracer != self._index_cloud:
                self._map_single(
                    tracers.quantity.data[:, :, :, i_tracer], pe1, pe2, self._qs
                )
        self._map_single_kord9(tracers.cloud, pe1, pe2, self._qs)

        if self._fill_negative_tracers:
            self._fillz(dp2, tracers)
