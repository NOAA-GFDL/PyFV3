import ndsl.constants as constants
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, exp
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval, log
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from pyfv3.tracers import FVTracers


from gt4py.cartesian.gtscript import __INLINED  # isort:skip


@gtfunction
def set_cappa(qvapor, cvm, r_vir):
    cappa = constants.RDGAS / (constants.RDGAS + cvm / (1.0 + r_vir * qvapor))
    return cappa


@gtfunction
def moist_cvm(qvapor, gz, ql, qs):
    # CK : GEOS applies the "max" function to tracer values
    cvm = (
        (1.0 - (max(qvapor, 0.0) + gz)) * constants.CV_AIR
        + max(qvapor, 0.0) * constants.CV_VAP
        + ql * constants.C_LIQ
        + qs * constants.C_ICE
    )
    return cvm


@gtfunction
def moist_cv_nwat0_fn():
    gz = 0
    cvm = constants.CV_AIR
    return cvm, gz


@gtfunction
def moist_cv_nwat6_fn(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
):
    # CK : GEOS applies the "max" function to tracer values
    ql = max(qliquid, 0.0) + max(qrain, 0.0)
    qs = max(qice, 0.0) + max(qsnow, 0.0) + max(qgraupel, 0.0)
    gz = ql + qs
    cvm = moist_cvm(qvapor, gz, ql, qs)
    return cvm, gz


@gtfunction
def moist_pt_func_nwat6(
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    q_con: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: Float,
):
    cvm, gz = moist_cv_nwat6_fn(
        qvapor, qliquid, qrain, qsnow, qice, qgraupel
    )  # if (nwat == 6) else moist_cv_default_fn(constants.CV_AIR)
    q_con = gz
    cappa = set_cappa(qvapor, cvm, r_vir)
    pt = pt * exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
    return cvm, gz, q_con, cappa, pt


@gtfunction
def moist_pt_func_nwat0(
    qvapor: FloatField,
    q_con: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: Float,
):
    cvm, gz = moist_cv_nwat0_fn()
    q_con = gz
    cappa = set_cappa(qvapor, cvm, r_vir)
    pt = pt * exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
    return cvm, gz, q_con, cappa, pt


@gtfunction
def last_pt(
    pt: FloatField,
    dtmp: Float,
    pkz: FloatField,
    gz: FloatField,
    qv: FloatField,
    zvir: Float,
):
    return (pt + dtmp * pkz) / ((1.0 + zvir * qv) * (1.0 - gz))


def moist_pt_last_step(
    tracers: FVTracers,
    pt: FloatField,
    pkz: FloatField,
    dtmp: Float,
    r_vir: Float,
):
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        pt (inout):
        pkz (in):
        dtmp (in):
        r_vir (in):
    """
    from __externals__ import i_graupel, i_ice, i_liquid, i_rain, i_snow, i_vapor, nwat

    with computation(PARALLEL), interval(...):
        if __INLINED(nwat == 0):
            _cvm, gz = moist_cv_nwat0_fn()
        elif __INLINED(nwat == 6):
            _cvm, gz = moist_cv_nwat6_fn(
                tracers.A[i_vapor],
                tracers.A[i_liquid],
                tracers.A[i_rain],
                tracers.A[i_snow],
                tracers.A[i_ice],
                tracers.A[i_graupel],
            )
        pt = last_pt(pt, dtmp, pkz, gz, tracers.A[i_vapor], r_vir)


@gtfunction
def compute_pkz_func(delp, delz, pt, cappa):
    # TODO use the exponential form for closer answer matching
    return exp(cappa * log(constants.RDG * delp / delz * pt))


def moist_pkz(
    tracers: FVTracers,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    r_vir: Float,
):
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        pkz (out):
        pt (in):
        cappa (out):
        delp (in):
        delz (in):
        r_vir (in):
    """
    from __externals__ import i_graupel, i_ice, i_liquid, i_rain, i_snow, i_vapor, nwat

    # TODO: What is happening with q_con and gz here?
    with computation(PARALLEL), interval(...):
        if __INLINED(nwat == 0):
            cvm, _gz = moist_cv_nwat0_fn()
        elif __INLINED(nwat == 6):
            cvm, _gz = moist_cv_nwat6_fn(
                tracers.A[i_vapor],
                tracers.A[i_liquid],
                tracers.A[i_rain],
                tracers.A[i_snow],
                tracers.A[i_ice],
                tracers.A[i_graupel],
            )

        cappa = set_cappa(tracers.A[i_vapor], cvm, r_vir)
        pkz = compute_pkz_func(delp, delz, pt, cappa)


def moist_te(
    tracers: FVTracers,
    u: FloatField,
    v: FloatField,
    w: FloatField,
    te: FloatFieldIJ,
    pt: FloatField,
    phis: FloatField,
    delp: FloatField,
    rsin2: FloatFieldIJ,
    cosa_s: FloatFieldIJ,
    hs: FloatFieldIJ,
    delz: FloatField,
    grav: Float,
):
    """
    Args:
        tracers (in):
        u (in):
        v (in):
        w (in):
        te (out):
        pt (in):
        phis (in):
        delp (in):
        rsin2 (in):
        cosa_s (in):
        hs (in):
    """
    from __externals__ import i_graupel, i_ice, i_liquid, i_rain, i_snow, i_vapor, nwat

    with computation(FORWARD), interval(-1, None):
        te = 0.0
        phis = hs
    with computation(BACKWARD), interval(0, -1):
        phis = phis[0, 0, 1] - grav * delz
    with computation(FORWARD), interval(0, -1):
        if __INLINED(nwat == 0):
            cvm, _gz = moist_cv_nwat0_fn()
        elif __INLINED(nwat == 6):
            cvm, _gz = moist_cv_nwat6_fn(
                tracers.A[i_vapor],
                tracers.A[i_liquid],
                tracers.A[i_rain],
                tracers.A[i_snow],
                tracers.A[i_ice],
                tracers.A[i_graupel],
            )

        te = te + delp * (
            cvm * pt
            + 0.5
            * (
                phis
                + phis[0, 0, 1]
                + w**2
                + 0.5
                * rsin2
                * (
                    u**2
                    + u[0, 1, 0] ** 2
                    + v**2
                    + v[1, 0, 0] ** 2
                    - (u + u[0, 1, 0]) * (v + v[1, 0, 0]) * cosa_s
                )
            )
        )


def te_zsum(
    te_2d: FloatFieldIJ,
    te0_2d: FloatFieldIJ,
    delp: FloatField,
    pkz: FloatField,
    zsum1: FloatFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            te_2d = te0_2d - te_2d
            zsum1 = pkz * delp

        with interval(1, None):
            zsum1 = zsum1 + pkz * delp


def cond_output(
    q_con: FloatField,
    tracers: FVTracers,
):
    from __externals__ import i_graupel, i_ice, i_liquid, i_rain, i_snow

    with computation(PARALLEL), interval(...):
        q_con = 0.0
        if __INLINED(i_liquid > 0):
            if tracers.A[i_liquid] > 0.0:
                q_con = q_con + tracers.A[i_liquid]
        if __INLINED(i_ice > 0):
            if tracers.A[i_ice] > 0.0:
                q_con = q_con + tracers.A[i_ice]
        if __INLINED(i_rain > 0):
            if tracers.A[i_rain] > 0.0:
                q_con = q_con + tracers.A[i_rain]
        if __INLINED(i_snow > 0):
            if tracers.A[i_snow] > 0.0:
                q_con = q_con + tracers.A[i_snow]
        if __INLINED(i_graupel > 0):
            if tracers.A[i_graupel] > 0.0:
                q_con = q_con + tracers.A[i_graupel]


def fv_setup(
    tracers: FVTracers,
    q_con: FloatField,
    cvm: FloatField,
    pkz: FloatField,
    pt: FloatField,
    cappa: FloatField,
    delp: FloatField,
    delz: FloatField,
    dp1: FloatField,
):
    """
    Args:
        qvapor (in):
        qliquid (in):
        qrain (in):
        qsnow (in):
        qice (in):
        qgraupel (in):
        q_con (out):
        cvm (out):
        pkz (out): p^(cappa)
        pt (in):
        cappa (out): Rd / Cp
        delp (in):
        delz (in):
        dp1 (out):
    """
    # without moist_cappa, we use a constant heat capacity for everything
    # variable heat capacity takes into account the mixing ratios of condensates
    # this is more accurate

    # TODO: what is being set up here, and how? update docstring
    with computation(PARALLEL), interval(...):
        from __externals__ import (
            i_graupel,
            i_ice,
            i_liquid,
            i_rain,
            i_snow,
            i_vapor,
            moist_phys,
            nwat,
        )

        if __INLINED(moist_phys):
            if __INLINED(nwat == 0):
                cvm, q_con = moist_cv_nwat0_fn()
            elif __INLINED(nwat == 6):
                cvm, q_con = moist_cv_nwat6_fn(
                    tracers.A[i_vapor],
                    tracers.A[i_liquid],
                    tracers.A[i_rain],
                    tracers.A[i_snow],
                    tracers.A[i_ice],
                    tracers.A[i_graupel],
                )  # if (nwat == 6) else moist_cv_default_fn(constants.CV_AIR)
            dp1 = constants.ZVIR * tracers.A[i_vapor]
            cappa = constants.RDGAS / (constants.RDGAS + cvm / (1.0 + dp1))
            pkz = exp(
                cappa
                * log(constants.RDG * delp * pt * (1.0 + dp1) * (1.0 - q_con) / delz)
            )
            # TODO: find documentation reference
            # (1.0 + dp1) * (1.0 - q_con) takes out condensate mass,
            # described in more detail in fv3 docs
            # condensates don't obey ideal gas law so they must be taken out
        else:
            dp1 = 0
            pkz = exp(constants.KAPPA * log(constants.RDG * delp * pt / delz))
            # cell mean pressure based on ideal gas law, raised to cappa
            # exponential log structure is faster on most processors
