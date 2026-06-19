from typing import Optional, Sequence

from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, interval
from ndsl.dsl.typing import (  # noqa: F401
    Bool,
    BoolField,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
)
from ndsl.stencils.basic_operations import copy
from pyfv3.stencils.remap_profile import RemapProfile


QMIN_DEFAULT = Float(0.0)
"""Minimum value allowed in a cell when remapping a field"""


def set_dp(dp1: FloatField, pe1: FloatField, lev: IntFieldIJ):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1
    with computation(FORWARD), interval(0, 1):
        lev = 0


def lagrangian_contributions(
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
    """
    Args:
        q (out):
        pe1 (in):
        pe2 (in):
        q4_1 (in):
        q4_2 (in):
        q4_3 (in):
        q4_4 (in):
        dp1 (in):
        lev (inout):
    """
    # TODO: Can we make lev a 2D temporary?
    with computation(FORWARD), interval(...):
        pl = (pe2 - pe1[0, 0, lev]) / dp1[0, 0, lev]
        if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
            pr = (pe2[0, 0, 1] - pe1[0, 0, lev]) / dp1[0, 0, lev]
            q = (
                q4_2[0, 0, lev]
                + 0.5
                * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                * (pr + pl)
                - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
            )
        else:
            qsum = (pe1[0, 0, lev + 1] - pe2) * (
                q4_2[0, 0, lev]
                + 0.5
                * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                * (1.0 + pl)
                - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
            )
            lev = lev + 1
            while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                lev = lev + 1
            dp = pe2[0, 0, 1] - pe1[0, 0, lev]
            esl = dp / dp1[0, 0, lev]
            qsum += dp * (
                q4_2[0, 0, lev]
                + 0.5
                * esl
                * (
                    q4_3[0, 0, lev]
                    - q4_2[0, 0, lev]
                    + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                )
            )
            q = qsum / (pe2[0, 0, 1] - pe2)
        lev = lev - 1


class LagrangianContribution:
    """Lagrangian contribution as it appears in FV3GFS/SHiELD"""

    def __init__(self, stencil_factory: StencilFactory, dims: Sequence[str]) -> None:
        self._lagrangian_contributions = stencil_factory.from_dims_halo(
            lagrangian_contributions,
            compute_dims=dims,
        )

    def __call__(
        self,
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
        self._lagrangian_contributions(
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


def lagrangian_contributions_interp(
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
    """
    Args:
        km (in):
        not_exit_loop (in/temp):
        LM1 (in/temp):
        LP0 (in/temp):
        q (in/out):
        pe1 (in):
        pe2 (in):
        q4_1 (in):
        q4_2 (in):
        q4_3 (in):
        q4_4 (in):
        dp1 (in):
        lev (inout):
    """

    # This computation creates a IntField that allows for "absolute" references
    # in the k-dimension for q and pe1.

    # INDEX_LM1 and INDEX_LP0 is initialized such that if it's plugged into "q"
    # (ex: q[0,0,INDEX_LM1]), the k level in q is "k = 0".

    # For example, during the stencil computation at k = 2, INDEX_LM1[i,j,2] = -2
    with computation(FORWARD):
        with interval(0, 1):
            INDEX_LM1 = 0
            INDEX_LP0 = 0
        with interval(1, None):
            INDEX_LM1 = INDEX_LM1[0, 0, -1] - 1
            INDEX_LP0 = INDEX_LP0[0, 0, -1] - 1

    # TODO: Can we make lev a 2D temporary?
    with computation(FORWARD), interval(...):
        LM1 = 1
        LP0 = 1
        not_exit_loop = True
        while LP0 <= km and not_exit_loop:
            if pe1[0, 0, INDEX_LP0] < pe2:
                LP0 = LP0 + 1
                INDEX_LP0 = INDEX_LP0 + 1
            else:
                not_exit_loop = False

        LM1 = max(LP0 - 1, 1)
        INDEX_LM1 = INDEX_LM1 + (LM1 - 1)
        LP0 = min(LP0, km)

        if LP0 == 1:
            INDEX_LP0 = INDEX_LM1
        elif LP0 <= km:
            INDEX_LP0 = INDEX_LM1 + 1
        else:
            INDEX_LP0 = INDEX_LM1

        if LM1 == 1 and LP0 == 1:
            q_temp = q[0, 0, INDEX_LM1] + (
                q[0, 0, INDEX_LM1 + 1] - q[0, 0, INDEX_LM1]
            ) * (pe2 - pe1[0, 0, INDEX_LM1]) / (
                pe1[0, 0, INDEX_LM1 + 1] - pe1[0, 0, INDEX_LM1]
            )

        elif LM1 == km and LP0 == km:
            q_temp = q[0, 0, INDEX_LM1] + (
                q[0, 0, INDEX_LM1] - q[0, 0, INDEX_LM1 - 1]
            ) * (pe2 - pe1[0, 0, INDEX_LM1]) / (
                pe1[0, 0, INDEX_LM1] - pe1[0, 0, INDEX_LM1 - 1]
            )

        elif LM1 == 1 or LP0 == km:
            q_temp = q[0, 0, INDEX_LP0] + (q[0, 0, INDEX_LM1] - q[0, 0, INDEX_LP0]) * (
                pe2 - pe1[0, 0, INDEX_LP0]
            ) / (pe1[0, 0, INDEX_LM1] - pe1[0, 0, INDEX_LP0])

        else:
            while pe2 < pe1[0, 0, lev] or pe2 > pe1[0, 0, lev + 1]:
                lev = lev + 1
            pl = (pe2 - pe1[0, 0, lev]) / dp1[0, 0, lev]
            if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
                pr = (pe2[0, 0, 1] - pe1[0, 0, lev]) / dp1[0, 0, lev]
                q_temp = (
                    q4_2[0, 0, lev]
                    + 0.5
                    * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                    * (pr + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
                )
            else:
                qsum = (pe1[0, 0, lev + 1] - pe2) * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                    * (1.0 + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
                )
                lev = lev + 1
                while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                    qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                    lev = lev + 1
                dp = pe2[0, 0, 1] - pe1[0, 0, lev]
                esl = dp / dp1[0, 0, lev]
                qsum += dp * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * esl
                    * (
                        q4_3[0, 0, lev]
                        - q4_2[0, 0, lev]
                        + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                    )
                )
                q_temp = qsum / (pe2[0, 0, 1] - pe2)

        lev = lev - 1

        q = q_temp


class LagrangianContributionInterpolated:
    """Lagrangian contribution as it appears in GEOS, modified from original
    FV3GFS version"""

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dims: Sequence[str],
    ) -> None:
        self._lagrangian_contributions_interp = stencil_factory.from_dims_halo(
            lagrangian_contributions_interp,
            compute_dims=dims,
        )

        self._INDEX_LM1 = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="",
            dtype=Int,
        )

        self._INDEX_LP0 = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            units="",
            dtype=Int,
        )
        self._km = stencil_factory.grid_indexing.domain[2]
        self._not_exit_loop = quantity_factory.zeros(
            [I_DIM, J_DIM], units="", dtype=bool
        )

    def __call__(
        self,
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
            km=self._km,
            not_exit_loop=self._not_exit_loop,
            INDEX_LM1=self._INDEX_LM1,
            INDEX_LP0=self._INDEX_LP0,
            q=q,
            pe1=pe1,
            pe2=pe2,
            q4_1=q4_1,
            q4_2=q4_2,
            q4_3=q4_3,
            q4_4=q4_4,
            dp1=dp1,
            lev=lev,
        )


class MapSingle(NDSLRuntime):
    """
    Fortran name is map_single, test classes are Map1_PPM_2d, Map_Scalar_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        kord: int,
        mode: int,
        dims: Sequence[str],
        interpolate_contribution: bool = False,
    ) -> None:
        super().__init__(stencil_factory)

        self._dp1 = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._q4_1 = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._q4_2 = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._q4_3 = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._q4_4 = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])
        self._lev = self.make_local(quantity_factory, [I_DIM, J_DIM], dtype=Int)

        # If the boundary condition is not given as an input, we use use a zero-reference
        # ⚠️ This _has_ to be a Quantity rather than a Local to be set to 0
        self._zero_qs = quantity_factory.zeros([I_DIM, J_DIM], "")
        self._zero_qs[:] = 0

        self._copy_stencil = stencil_factory.from_dims_halo(
            copy,
            compute_dims=dims,
        )

        self._set_dp = stencil_factory.from_dims_halo(
            set_dp,
            compute_dims=dims,
        )

        self._remap_profile = RemapProfile(
            stencil_factory,
            quantity_factory,
            kord,
            mode,
            dims=dims,
        )

        self._lagrangian_contributions = (
            LagrangianContributionInterpolated(stencil_factory, quantity_factory, dims)
            if interpolate_contribution
            else LagrangianContribution(stencil_factory, dims)
        )

    def __call__(
        self,
        q1: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        qmin: Float,
        qs: Optional[FloatFieldIJ] = None,
    ) -> None:
        """
        Compute x-flux using the PPM method.

        Args:
            q1 (out): Remapped field on Eulerian grid
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            qmin (in): Minimum allowed value of the remapped field
            qs (in): Bottom boundary condition
        """

        self._copy_stencil(q1, self._q4_1)
        self._set_dp(self._dp1, pe1, self._lev)

        if qs is None:
            self._remap_profile(
                self._zero_qs,
                self._q4_1,
                self._q4_2,
                self._q4_3,
                self._q4_4,
                self._dp1,
                qmin,
            )
        else:
            self._remap_profile(
                qs,
                self._q4_1,
                self._q4_2,
                self._q4_3,
                self._q4_4,
                self._dp1,
                qmin,
            )

        self._lagrangian_contributions(
            q=q1,
            pe1=pe1,
            pe2=pe2,
            q4_1=self._q4_1,
            q4_2=self._q4_2,
            q4_3=self._q4_3,
            q4_4=self._q4_4,
            dp1=self._dp1,
            lev=self._lev,
        )
