"""
Cascaded Quadrotor Controller:
Combines Outer-Loop PID Position Controller with Inner-Loop Paper 1 Quaternion Attitude Controller.
Computes world-frame thrust and body-frame control torques for 6-DoF trajectory tracking.
"""

import numpy as np
from src.quaternion_math import quat_normalize, quat_to_rot_matrix
from src.quaternion_controller import NonlinearP2QuaternionController

class CascadedQuadrotorTracker:
    def __init__(self, mass: float = 0.5, gravity: float = 9.81, dt: float = 0.005):
        self.m = mass
        self.g = gravity
        self.dt = dt

        # Outer-loop Position PID Gains.
        # Tuned against the wind-loaded mission (see tests/test_wind_rejection.py).
        # The previous values (Kp=[2,2,3], Kd=[1.5,1.5,2], Ki=[0.15,0.15,0.2])
        # gave omega_n = sqrt(2) = 1.41 rad/s at zeta = 0.53 -- too slow to hold
        # the 2 m/s reference (0.62 m of pure along-track lag, with the
        # acceleration clip saturated 37% of the transit) and too lightly damped
        # (ground speed overshot the 2 m/s reference to 3.1 m/s). These give
        # omega_n = sqrt(6) = 2.45 rad/s at zeta = 0.94: near-critically damped,
        # and still ~6x slower than the inner attitude loop's 15 rad/s, which
        # preserves the timescale separation the cascade depends on.
        self.Kp_pos = np.array([6.0, 6.0, 8.0])
        self.Kd_pos = np.array([4.6, 4.6, 5.6])
        self.Ki_pos = np.array([1.5, 1.5, 2.0])

        # Integral authority. A steady 3.5 m/s crosswind on this airframe is a
        # ~0.21 N per-axis drag force, i.e. ~0.6 m/s^2 the loop must hold
        # indefinitely. The integral can supply at most Ki * integral_clamp;
        # at the old Ki=0.15 / clamp=1.0 that ceiling was 0.15 m/s^2, so the
        # balance had to come from the proportional term -- a permanent 0.7 m
        # position offset at the goal. Ki=1.5 with clamp=2.0 covers the steady
        # wind with margin. The clamp is the anti-windup bound.
        self.integral_clamp = 2.0

        # Acceleration bounds for tilt safety (see compute_control).
        self.accel_limit_transit = 3.0
        self.accel_limit_hold = 6.0

        self.e_p_integral = np.zeros(3)

        # Inner-loop Paper 1 Quaternion Controller.
        # Pq=20, Pw=4 (the paper's nominal gains) are wildly oversized for this
        # airframe's actual inertia (Ixx=Iyy=6.5e-4 kg*m^2): the closed-loop
        # attitude poles land near +/-6150 rad/s, which needs dt < ~3e-4s for
        # stable explicit integration -- 15x finer than this sim's dt=0.005s.
        # At the real dt that shows up as a permanent limit-cycle oscillation
        # (torque flips sign every single step) that the outer position loop can
        # never settle through, so the drone orbits the target instead of landing
        # on it. These gains place the attitude closed-loop poles at ~15 rad/s
        # (critically damped), which is comfortably stable at dt=0.005s while
        # still responding ~10x faster than the outer position loop.
        self.att_controller = NonlinearP2QuaternionController(
            Pq=0.15,
            Pw=0.025,
            max_torque=4.0,
            noise_amp=0.0
        )

    def reset(self):
        self.e_p_integral = np.zeros(3)

    def compute_control(self, p_target: np.ndarray, v_target: np.ndarray, a_target: np.ndarray, state: dict) -> tuple:
        """
        Computes control torque tau and scalar thrust force F_thrust.
        """
        p_curr = state['pos']
        v_curr = state['vel']
        q_curr = state['quat']
        # MuJoCo reports a freejoint's angular velocity in the BODY frame, which
        # is already the frame the P^2 attitude law needs -- no rotation here.
        w_body = state['omega']

        # Position and Velocity Errors
        e_p = p_target - p_curr
        e_v = v_target - v_curr

        self.e_p_integral += e_p * self.dt
        self.e_p_integral = np.clip(self.e_p_integral, -self.integral_clamp, self.integral_clamp)

        # Desired Translational Acceleration Command
        a_des = a_target + self.Kp_pos * e_p + self.Kd_pos * e_v + self.Ki_pos * self.e_p_integral

        # Acceleration Bounds for Tilt Safety.
        # While actively following the trajectory (nonzero feedforward), keep the
        # bound tight so tilt stays modest during fast transit (3.0 m/s^2 is a
        # ~17 deg steady tilt; measured peak including transients is ~25 deg). Once the reference
        # has arrived and stopped (v_target/a_target ~ 0, i.e. holding at the goal),
        # there is no feedforward competing for clip budget, so a wider bound is
        # safe and gives the controller enough authority to actually brake and
        # settle on the target instead of coasting/overshooting past it.
        is_holding = np.linalg.norm(v_target) < 1e-3 and np.linalg.norm(a_target) < 1e-3
        accel_limit = self.accel_limit_hold if is_holding else self.accel_limit_transit
        a_des = np.clip(a_des, -accel_limit, accel_limit)

        # Desired World Thrust Vector: F_world = m * (a_des + g * z_hat)
        F_world = self.m * (a_des + np.array([0.0, 0.0, self.g]))

        R_curr = quat_to_rot_matrix(q_curr)
        cos_tilt = max(0.3, R_curr[2, 2])

        # Total Thrust Force magnitude needed to produce the full desired
        # acceleration vector (not just its vertical component) along the
        # body z-axis, compensated for tilt angle. Using only a_des[2] here
        # under-supplies thrust whenever lateral acceleration is significant,
        # which starves the vehicle of the acceleration the position loop
        # commanded and shows up as poor tracking / oscillation.
        thrust_force = np.linalg.norm(F_world) / cos_tilt
        thrust_force = np.clip(thrust_force, 0.1, 12.0)

        # Desired Body Z-Axis Direction
        z_b_des = F_world / np.linalg.norm(F_world)

        x_c = np.array([1.0, 0.0, 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        if np.linalg.norm(y_b_des) < 1e-6:
            y_b_des = np.array([0.0, 1.0, 0.0])
        else:
            y_b_des /= np.linalg.norm(y_b_des)

        x_b_des = np.cross(y_b_des, z_b_des)
        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])

        # Convert R_des to desired quaternion q_ref
        tr = np.trace(R_des)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            q0 = 0.25 * S
            q1 = (R_des[2, 1] - R_des[1, 2]) / S
            q2 = (R_des[0, 2] - R_des[2, 0]) / S
            q3 = (R_des[1, 0] - R_des[0, 1]) / S
        else:
            q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0
        q_ref = quat_normalize(np.array([q0, q1, q2, q3]))

        # Compute body control torque via Paper 1 Quaternion Attitude Controller.
        # Both inputs are body-frame (q_err from compute_quat_error is expressed
        # in the body frame of q_curr, and w_body comes straight off the
        # freejoint), so the result is a body-frame torque.
        tau_body, _, _ = self.att_controller.compute_torque(q_ref, q_curr, w_body, add_noise=False)

        # Returned as-is: qfrc_applied[3:6] on a freejoint is BODY frame, so
        # rotating this into the world frame first (as an earlier version did)
        # applied the attitude-error term about the wrong axes. The effect was
        # small at the modest tilts this controller commands, and vanished
        # entirely for single-axis recoveries -- where R shares its axis with
        # the error and so leaves it invariant -- but it is still wrong, and it
        # grows with commanded tilt.
        return tau_body, thrust_force
