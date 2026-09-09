"""
Regression tests for how control and disturbance wrenches reach MuJoCo.

Two defects motivated these tests:

1. External disturbance forces were silently discarded. Callers wrote the wind
   force into `data.qfrc_applied[0:3]` and then called `sim.step(...)`, which
   *assigned* the thrust into that same slot -- overwriting the wind before
   `mj_step` ever saw it.

2. Body/world frame mix-up on the free joint. For a MuJoCo `freejoint`,
   `qvel[3:6]` and `qfrc_applied[3:6]` are both expressed in the BODY frame
   (verified by `test_freejoint_rotational_dofs_are_body_frame` below). The
   cascaded tracker treated them as world-frame, rotating omega by R^T on the
   way in and the torque by R on the way out. Both rotations are spurious; they
   vanish at zero tilt/yaw, which is why the bug hid.
"""

import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quadrotor_mujoco import QuadrotorMuJoCoSim
from src.quaternion_math import quat_to_rot_matrix
from src.trajectory_tracker import CascadedQuadrotorTracker

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "quadrotor.xml")


def yaw_quat(angle_rad):
    return np.array([np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)])


def test_freejoint_rotational_dofs_are_body_frame():
    """Pins the MuJoCo convention the rest of these tests (and the fix) rely on.

    A torque on the second rotational DOF must accelerate about the BODY y-axis
    (using Iyy), regardless of how the body is oriented in the world.
    """
    xml = """
    <mujoco><option gravity="0 0 0" timestep="0.001"/>
    <worldbody><body name="b"><freejoint/>
    <inertial pos="0 0 0" mass="1" fullinertia="1e-3 2e-3 3e-3 0 0 0"/>
    <geom type="sphere" size="0.05"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    # +90 deg about world x, so body y maps to world z and body z to world -y.
    d.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    mujoco.mj_forward(m, d)

    d.qfrc_applied[4] = 1.0
    for _ in range(10):
        mujoco.mj_step(m, d)

    # Body frame -> uses Iyy=2e-3 -> alpha=500 rad/s^2 -> 5.0 rad/s after 10ms.
    # World frame would have used Izz=3e-3 -> 3.33 rad/s.
    assert d.qvel[4] == pytest.approx(5.0, rel=1e-3)
    assert np.allclose(d.qvel[[3, 5]], 0.0, atol=1e-9)


def test_external_force_is_not_overwritten_by_thrust():
    """A disturbance force passed to step() must actually act on the vehicle."""
    sim = QuadrotorMuJoCoSim(xml_path=MODEL, gui=False)
    sim.reset(pos=np.array([0.0, 0.0, 5.0]))

    # Exactly cancel gravity with thrust so any x-motion comes from the push.
    hover = 0.5 * 9.81
    push = np.array([1.0, 0.0, 0.0])

    for _ in range(250):  # 0.5 s
        sim.step(tau=np.zeros(3), thrust_force=hover, f_ext_world=push)

    # a = F/m = 2 m/s^2 for 0.5 s -> vx = 1.0 m/s, x = 0.25 m.
    state = sim.get_state()
    assert state['vel'][0] == pytest.approx(1.0, rel=0.02)
    assert state['pos'][0] == pytest.approx(0.25, rel=0.05)
    assert state['pos'][2] == pytest.approx(5.0, abs=1e-3)


def test_step_applies_torque_in_body_frame():
    """Torque handed to step() is body-frame: yawing the vehicle must not
    change which body axis a given torque command spins up."""
    sim = QuadrotorMuJoCoSim(xml_path=MODEL, gui=False)
    sim.reset(pos=np.array([0.0, 0.0, 5.0]), quat=yaw_quat(np.pi / 2))

    tau_body = np.array([0.02, 0.0, 0.0])  # pure body roll
    for _ in range(50):
        sim.step(tau=tau_body, thrust_force=0.0)

    omega_body = sim.get_state()['omega']
    assert omega_body[0] > 0.0
    assert abs(omega_body[1]) < 1e-6 * max(abs(omega_body[0]), 1.0)

    # And in the world frame that body roll shows up about world +y, because
    # the vehicle is yawed 90 deg.
    R = quat_to_rot_matrix(sim.get_state()['quat'])
    omega_world = R @ omega_body
    assert omega_world[1] > 0.0
    assert abs(omega_world[0]) < 0.05 * omega_world[1]


def test_tracker_consumes_and_returns_body_frame_quantities():
    """The cascade must hand the freejoint's body-frame omega to the P^2 law
    unrotated, and return that law's body-frame torque unrotated.

    This is asserted white-box because no closed-loop test discriminates it at
    the current gains: the position loop clips acceleration to 2 m/s^2, so
    commanded tilt stays under ~11 deg (R ~ I), and in a single-axis recovery
    R shares its rotation axis with the attitude error and therefore leaves it
    invariant. The defect is real but latent -- it grows with commanded tilt,
    so it would surface the moment the acceleration bound is raised.
    """
    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=0.002)

    # Tilted AND yawed, with nonzero body rates, so any stray rotation shows.
    roll, pitch, yaw = np.radians(35.0), np.radians(-20.0), np.radians(70.0)
    from src.quaternion_math import euler_to_quat
    state = {
        'pos': np.array([0.4, -0.3, 4.7]),
        'vel': np.array([0.2, 0.1, -0.05]),
        'quat': euler_to_quat(roll, pitch, yaw),
        'omega': np.array([0.30, -0.20, 0.10]),   # body frame
        'time': 1.0,
    }

    seen = {}
    inner = tracker.att_controller
    original = inner.compute_torque

    def spy(q_ref, q_m, omega, add_noise=True):
        seen['omega_in'] = np.array(omega, dtype=np.float64)
        tau, q_err, axis_err = original(q_ref, q_m, omega, add_noise)
        seen['tau_out'] = np.array(tau, dtype=np.float64)
        return tau, q_err, axis_err

    inner.compute_torque = spy
    try:
        tau_returned, _ = tracker.compute_control(
            np.array([0.0, 0.0, 5.0]), np.zeros(3), np.zeros(3), state)
    finally:
        inner.compute_torque = original

    np.testing.assert_allclose(seen['omega_in'], state['omega'], atol=1e-12,
                               err_msg="omega was rotated before the P^2 law")
    np.testing.assert_allclose(tau_returned, seen['tau_out'], atol=1e-12,
                               err_msg="torque was rotated after the P^2 law")


def _fly_to_hold(yaw_rad, steps=6000):
    """Holds station at a fixed target from a 1 m offset at the given yaw.
    Returns the final position error in metres."""
    sim = QuadrotorMuJoCoSim(xml_path=MODEL, gui=False)
    tracker = CascadedQuadrotorTracker(mass=0.5, gravity=9.81, dt=sim.dt)

    target = np.array([0.0, 0.0, 5.0])
    sim.reset(pos=target + np.array([1.0, 0.0, 0.0]), quat=yaw_quat(yaw_rad))
    tracker.reset()

    zero = np.zeros(3)
    for _ in range(steps):
        state = sim.get_state()
        tau, thrust = tracker.compute_control(target, zero, zero, state)
        sim.step(tau=tau, thrust_force=thrust)

    return float(np.linalg.norm(sim.get_state()['pos'] - target))


def test_hold_station_is_yaw_invariant():
    """Station-keeping must be as good yawed as aligned."""
    err_aligned = _fly_to_hold(0.0)
    err_yawed = _fly_to_hold(np.pi / 2)

    assert err_aligned < 0.05, f"aligned hold error {err_aligned:.3f} m"
    assert err_yawed < 0.05, f"yawed hold error {err_yawed:.3f} m"
