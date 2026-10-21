import numpy as np
from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from attitude_control.position_controller import PositionController
from .trajectory_converter import TrajectoryConverter
from attitude_control.quaternion import Quaternion

class MergedSimulation:
    """
    Merged simulation — Paper 1 path planning + Paper 2 attitude control + Outer-loop position control.

    The B-Spline path generates target 3D positions and path tangents.
    The outer-loop PositionController converts 3D position errors into desired thrust & quaternion.
    The inner-loop quaternion P² controller tracks those attitudes.
    The full 6-DOF quadrotor dynamics integrated over time updates actual 3D position & velocity.
    """

    def __init__(self, spline, dt=0.01, total_time=20.0,
                 Pq=20.0, Pw=4.0, Kp_pos=(2.0, 2.0, 4.0), Kd_pos=(1.5, 1.5, 2.5),
                 enable_motor_model=False):
        self.spline          = spline
        self.dt              = dt
        self.total_time      = total_time
        self.Pq              = Pq
        self.Pw              = Pw

        self.drone           = QuadrotorModel(enable_motor_model=enable_motor_model)
        self.controller      = P2Controller(Pq=Pq, Pw=Pw)
        self.pos_controller  = PositionController(Kp_pos=Kp_pos, Kd_pos=Kd_pos, mass=self.drone.mass, g=self.drone.g)
        self.converter       = TrajectoryConverter()

        self.results = {
            't':            [],
            'position':     [],
            'position_ref': [],
            'velocity':     [],
            'euler':        [],
            'euler_ref':    [],
            'q_measured':   [],
            'q_ref':        [],
            'axis_error':   [],
            'torque':       [],
            'thrust':       [],
            'omega':        [],
            'u_param':      [],
            'err_mag':      [],
            'pos_err_mag':  []
        }

    def run(self, noise_std=0.1, error_threshold=0.05, u_step=0.005, mode='6dof'):
        """
        Run 3D trajectory tracking simulation.
        mode='6dof': Full 6-DOF position + attitude integration.
        mode='attitude_only': Legacy attitude-only integration.
        """
        n_steps = int(self.total_time / self.dt)

        u_current = 0.001
        pos_start, q_ref_start = self.converter.get_reference_at(self.spline, u_current)

        # Initialize drone position and attitude aligned with path start
        self.drone.reset(position=pos_start)
        self.drone.q = Quaternion(*q_ref_start.q)
        q_ref_current = q_ref_start

        print(f"Running Merged Simulation ({mode.upper()} mode) for {self.total_time}s...")

        for i in range(n_steps):
            t = i * self.dt

            pos_ref, q_ref_path = self.converter.get_reference_at(self.spline, u_current)
            tangent = self.spline.evaluate_derivative(u_current, 1)
            yaw_ref = np.arctan2(tangent[1], tangent[0]) if np.hypot(tangent[0], tangent[1]) > 1e-3 else 0.0

            omega_noisy = self.drone.omega + np.random.normal(0, noise_std, 3)

            if mode == '6dof':
                # Outer loop: Position -> Total Thrust & Desired Quaternion
                thrust, q_ref_des = self.pos_controller.compute(
                    pos_ref, self.drone.position,
                    vel_ref=np.zeros(3), vel=self.drone.velocity,
                    yaw_ref=yaw_ref
                )
                q_ref_current = q_ref_des
            else:
                thrust = self.drone.mass * self.drone.g
                q_ref_current = q_ref_path

            # Inner loop: Quaternion P² Attitude Control
            q_err = self.controller.compute_error(q_ref_current, self.drone.q)
            err_mag = np.linalg.norm([q_err.q1, q_err.q2, q_err.q3])
            pos_err_mag = np.linalg.norm(pos_ref - self.drone.position)

            # Advance path progress when position and/or attitude converge
            if mode == '6dof':
                if pos_err_mag < 25.0 and u_current < 0.99:
                    u_current = min(u_current + u_step, 0.99)
            else:
                if err_mag < error_threshold and u_current < 0.99:
                    u_current = min(u_current + u_step, 0.99)

            tau, axis_err = self.controller.compute_torque(
                q_ref_current, self.drone.q, omega_noisy
            )

            # Step physics
            if mode == '6dof':
                self.drone.step_6dof(tau, total_thrust=thrust, dt=self.dt)
            else:
                self.drone.step_attitude_only(tau, self.dt)

            # Logging
            self.results['t'].append(t)
            self.results['position'].append(self.drone.position.copy())
            self.results['position_ref'].append(pos_ref.copy())
            self.results['velocity'].append(self.drone.velocity.copy())
            self.results['euler'].append(self.drone.q.to_euler())
            self.results['euler_ref'].append(q_ref_current.to_euler())
            self.results['q_measured'].append(self.drone.q.q.copy())
            self.results['q_ref'].append(q_ref_current.q.copy())
            self.results['axis_error'].append(axis_err.copy())
            self.results['torque'].append(tau.copy())
            self.results['thrust'].append(thrust)
            self.results['omega'].append(self.drone.omega.copy())
            self.results['u_param'].append(u_current)
            self.results['err_mag'].append(err_mag)
            self.results['pos_err_mag'].append(pos_err_mag)

            # Progress print
            if i % max(1, n_steps // 10) == 0:
                pct = 100 * i // n_steps
                print(f"  {pct:3d}% | t={t:5.1f}s | u={u_current:.3f} | "
                      f"PosErr={pos_err_mag:6.1f}m | AttErr={err_mag:.4f} | "
                      f"Pos=[{self.drone.position[0]:.1f}, {self.drone.position[1]:.1f}, {self.drone.position[2]:.1f}]m")

        print("Simulation complete.")
        return self.results

