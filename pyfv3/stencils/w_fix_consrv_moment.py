from gt4py.cartesian.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

from ndsl.dsl.typing import BoolFieldIJ, Float, FloatField, FloatFieldIJ


def W_fix_consrv_moment(
    w: FloatField,
    w2: FloatField,
    dp2: FloatField,
    gz: FloatFieldIJ,
    w_max: Float,
    w_min: Float,
    compute_performed: BoolFieldIJ,
):
    """
    Args:
        w (in/out):
        w2 (in?):
        dp2(in):
        w_max(in):
        w_min(in):
        compute_performed: (Internal Temporary),
    """

    with computation(PARALLEL), interval(...):
        w2 = w

    with computation(FORWARD):
        with interval(0, 1):
            compute_performed = False
            if w2 > w_max:
                gz = (w2 - w_max) * dp2
                w2 = w_max
                compute_performed = True
            elif w2 < w_min:
                gz = (w2 - w_min) * dp2
                w2 = w_min
                compute_performed = True
        with interval(1, -1):
            if compute_performed:
                w2 = w2 + gz / dp2
                compute_performed = False
            if w2 > w_max:
                gz = (w2 - w_max) * dp2
                w2 = w_max
                compute_performed = True
            elif w2 < w_min:
                gz = (w2 - w_min) * dp2
                w2 = w_min
                compute_performed = True

    with computation(BACKWARD):
        with interval(-1, None):
            compute_performed = False
            if w2 > w_max:
                gz = (w2 - w_max) * dp2
                w2 = w_max
                compute_performed = True
            elif w2 < w_min:
                gz = (w2 - w_min) * dp2
                w2 = w_min
                compute_performed = True
        with interval(1, -1):
            if compute_performed:
                w2 = w2 + gz / dp2
                compute_performed = False
            if w2 > w_max:
                gz = (w2 - w_max) * dp2
                w2 = w_max
                compute_performed = True
            elif w2 < w_min:
                gz = (w2 - w_min) * dp2
                w2 = w_min
                compute_performed = True

    with computation(FORWARD), interval(0, 1):
        if w2 > (w_max * 2.0):
            w2 = w_max * 2.0
        elif w2 < (w_min * 2.0):
            w2 = w_min * 2.0

    with computation(PARALLEL), interval(...):
        w = w2
