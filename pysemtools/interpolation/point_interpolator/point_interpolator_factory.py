"""Factory for selection of point interpolator"""


def get_point_interpolator(interpolator_type, *args, **kwargs):
    """Get point interpolator"""
    if interpolator_type == "single_point_legendre":
        from .single_point_legendre_interpolator import LegendreInterpolator as sp_leg

        n = args[0]
        return sp_leg(n)
    elif interpolator_type == "single_point_lagrange":
        from .single_point_lagrange_interpolator import LagrangeInterpolator as sp_lag

        n = args[0]
        return sp_lag(n)
    elif interpolator_type == "multiple_point_legendre_numpy":
        from .multiple_point_interpolator_legendre_numpy import (
            LegendreInterpolator as mp_leg_numpy,
        )

        n = args[0]
        max_pts = kwargs.get("max_pts", 128)
        max_elems = kwargs.get("max_elems", 1)
        return mp_leg_numpy(n, max_pts=max_pts, max_elems=max_elems)
    elif interpolator_type == "multiple_point_legendre_torch":
        from .multiple_point_interpolator_legendre_torch import (
            LegendreInterpolator as mp_leg_torch,
        )

        n = args[0]
        max_pts = kwargs.get("max_pts", 1000)
        max_elems = kwargs.get("max_elems", 1)
        use_autograd = kwargs.get("use_autograd", False)
        return mp_leg_torch(
            n, max_pts=max_pts, max_elems=max_elems, use_autograd=use_autograd
        )
    else:
        raise ValueError("Invalid interpolator type")
