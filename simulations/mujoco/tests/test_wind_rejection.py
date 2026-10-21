"""
Closed-loop wind-rejection tests for the cascaded tracker.

These pin the flight quality that the gain retune bought, so it cannot silently
regress. They fly a straight, feasible reference in the real MuJoCo scene under
the same wind model the mission uses.

Reference numbers on the full mission path, averaged over wind speeds
{0, 3.5, 6, 9} m/s and three gust seeds:

                       mean transit error    final arrival error
    before retune              1.312 m               0.603 m
    after  retune              0.227 m               0.006 m
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quadrotor_mujoco import QuadrotorMuJoCoSim
from src.quaternion_math import quat_to_rot_matrix
from src.time_parameterization import build_speed_limited_profile
from src.trajectory_tracker import CascadedQuadrotorTracker
from src.wind_model import WindDisturbanceModel

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "airspace_scene.xml")


class _StraightPath:
    """Minimal evaluator matching the BSplineEvaluator interface used by
    build_speed_limited_profile: a straight line from `a` to `b` over u in [0,1]."""

    def __init__(self, a, b):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)

    def eval_pos(self, u):
        u = np.asarray(u, dtype=np.float64)
        if u.ndim == 0:
            return self.a + u * (self.b - self.a)
        return self.a + u[:, None] * (self.b - self.a)

    def eval_vel(self, u):
        return self.b - self.a


def _fly(wind_speed, seed, hold_time=8.0):
    """Flies a 12 m straight leg under wind, then holds at the goal.
    Returns (mean transit error, final error, max tilt in degrees)."""
    np.random.seed(seed)

    sim = QuadrotorMuJoCoSim(xml_path=MODEL, gui=False)
    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=sim.dt)
    wind = WindDisturbanceModel(wind_speed=wind_speed, wind_direction_deg=45.0,
                                gust_intensity=0.5, dt=sim.dt)

    path = _StraightPath([0.0, 0.0, 3.0], [10.0, 6.0, 5.0])
    eval_at_time, t_path = build_speed_limited_profile(path, 1.0, v_max=2.0, a_max=1.2)
    goal = path.eval_pos(1.0)

    sim.reset(pos=path.eval_pos(0.0), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    tracker.reset()

    errors, tilts = [], []
    n_total = int((t_path + hold_time) / sim.dt)
    n_transit = int(t_path / sim.dt)

    for _ in range(n_total):
        state = sim.get_state()
        t = state['time']
        if t <= t_path:
            p_ref, v_ref, a_ref = eval_at_time(t)
        else:
            p_ref, v_ref, a_ref = goal.copy(), np.zeros(3), np.zeros(3)

        R = quat_to_rot_matrix(state['quat'])
        f_wind, tau_wind = wind.compute_disturbance(state['vel'], R)
        tau, thrust = tracker.compute_control(p_ref, v_ref, a_ref, state)
        sim.step(tau=tau, thrust_force=thrust,
                 f_ext_world=f_wind, tau_ext_body=tau_wind)

        errors.append(np.linalg.norm(state['pos'] - p_ref))
        tilts.append(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))

    errors = np.array(errors)
    return errors[:n_transit].mean(), errors[-1], max(tilts)


@pytest.mark.parametrize("wind_speed", [0.0, 3.5, 6.0])
def test_tracks_and_settles_under_wind(wind_speed):
    mean_err, final_err, max_tilt = _fly(wind_speed, seed=0)

    assert mean_err < 0.35, f"transit tracking {mean_err:.3f} m at {wind_speed} m/s"
    assert final_err < 0.10, f"did not settle on goal: {final_err:.3f} m at {wind_speed} m/s"
    assert max_tilt < 40.0, f"excessive tilt {max_tilt:.1f} deg at {wind_speed} m/s"


def test_steady_wind_leaves_no_standing_offset():
    """The integral term must absorb the steady crosswind. Without enough
    integral authority the proportional term has to hold the wind, which shows
    up as a permanent position offset at the goal."""
    _, final_calm, _ = _fly(0.0, seed=0)
    _, final_windy, _ = _fly(3.5, seed=0)

    assert final_windy < 0.10, f"standing offset under wind: {final_windy:.3f} m"
    assert final_windy - final_calm < 0.08, (
        f"wind added {final_windy - final_calm:.3f} m of standing offset")


def test_gust_response_is_repeatable_across_seeds():
    """Different gust realisations must not change flight quality much."""
    results = [_fly(3.5, seed=s) for s in (0, 1, 2)]
    means = np.array([r[0] for r in results])
    finals = np.array([r[1] for r in results])

    assert means.std() < 0.05, f"transit error varies with gust seed: {means}"
    assert finals.max() < 0.10, f"a gust seed prevented settling: {finals}"
