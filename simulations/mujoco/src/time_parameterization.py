"""
Trajectory Time-Parameterization

Turns a geometric path (a B-spline parameterized by u in [0,1]) into a timed
reference the position controller can actually follow.

A naive quintic map in u ignores how unevenly the GA-optimized knot spacing
distributes arc length over u, so it demands peak accelerations far beyond what
the cascade's tilt-safe bound can deliver -- measured at 8.6 m/s^2 against a
2.0 m/s^2 clip on the default mission, which alone produced ~0.57 m of mean
tracking error with no wind at all. Re-timing against arc length with explicit
speed and acceleration bounds removes that error source and leaves the
controller's authority free for actually rejecting disturbances.
"""

import numpy as np


def build_speed_limited_profile(evaluator, scale: float, v_max: float, a_max: float, n_samples: int = 4000):
    """
    Re-times the geometric B-spline path (parameterized by u in [0,1]) using a
    trapezoidal speed profile bounded by v_max/a_max, instead of a fixed-duration
    quintic map. A quintic-in-u map ignores how unevenly the GA-optimized knot
    spacing distributes arc length over u, so it can demand instantaneous speeds
    (~9-13 m/s here) far beyond what the position controller's tilt-safe
    acceleration bound can brake from -- the drone flies through the goal instead
    of stopping there. This profile guarantees |v_ref| <= v_max and decelerates
    to exactly zero at the end of the path, so the drone can actually settle on
    the target.

    Returns (eval_at_time, T_total) where eval_at_time(t) -> (p_ref, v_ref, a_ref)
    in scaled world coordinates, and T_total is the profile's duration in seconds.
    """
    u_samples = np.linspace(0.0, 1.0, n_samples)
    pts = evaluator.eval_pos(u_samples) * scale
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc_len = np.insert(np.cumsum(seg_len), 0, 0.0)
    L_total = arc_len[-1]

    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if 2.0 * d_acc >= L_total:
        # Path too short to reach v_max: triangular profile.
        t_acc = np.sqrt(L_total / a_max) if a_max > 0 else 0.0
        v_peak = a_max * t_acc
        d_acc = 0.5 * L_total
        t_cruise = 0.0
    else:
        v_peak = v_max
        t_cruise = (L_total - 2.0 * d_acc) / v_max

    T_total = 2.0 * t_acc + t_cruise

    def dist_speed_accel(t: float):
        if t <= 0.0:
            return 0.0, 0.0, a_max
        if t < t_acc:
            return 0.5 * a_max * t**2, a_max * t, a_max
        if t < t_acc + t_cruise:
            return d_acc + v_peak * (t - t_acc), v_peak, 0.0
        if t < T_total:
            t_dec = t - (t_acc + t_cruise)
            d = d_acc + v_peak * t_cruise + v_peak * t_dec - 0.5 * a_max * t_dec**2
            return d, max(v_peak - a_max * t_dec, 0.0), -a_max
        return L_total, 0.0, 0.0

    def eval_at_time(t: float):
        d, v, a_tan = dist_speed_accel(t)
        u = float(np.interp(d, arc_len, u_samples))
        p_ref = evaluator.eval_pos(u) * scale
        tangent = evaluator.eval_vel(u) * scale
        tnorm = np.linalg.norm(tangent)
        tangent_unit = tangent / tnorm if tnorm > 1e-9 else np.zeros(3)
        v_ref = v * tangent_unit
        a_ref = a_tan * tangent_unit
        return p_ref, v_ref, a_ref

    return eval_at_time, T_total
