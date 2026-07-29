import numpy as np
from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from .trajectory_converter import TrajectoryConverter
from attitude_control.quaternion import Quaternion

class MergedSimulation:
    """
    Merged simulation — Paper 1 feeds references into Paper 2.

    The B-Spline path generates a sequence of desired attitudes.
    The quaternion P² controller tracks those attitudes.
    Path progress is error-gated: the path parameter only advances
    once the current attitude error has converged below a threshold.
    """

    def __init__(self, spline, dt=0.01, total_time=20.0,
                 Pq=20.0, Pw=4.0):
        self.spline      = spline
        self.dt          = dt
        self.total_time  = total_time
        self.Pq          = Pq
        self.Pw          = Pw

        self.drone       = QuadrotorModel()
        self.controller  = P2Controller(Pq=Pq, Pw=Pw)
        self.converter   = TrajectoryConverter()

        self.results = {
            't':          [],
            'euler':      [],
            'euler_ref':  [],
            'q_measured': [],
            'q_ref':      [],
            'axis_error': [],
            'torque':     [],
            'omega':      [],
            'u_param':    [],
            'err_mag':    []
        }

    def run(self, noise_std=0.1, error_threshold=0.05, u_step=0.005):
        """
        Run attitude tracking simulation with error-gated path progress.
        """
        n_steps = int(self.total_time / self.dt)

        u_current = 0.001
        _, q_ref = self.converter.get_reference_at(
            self.spline, u_current
        )

        # Initialize drone — start aligned with path start
        self.drone.q = Quaternion(*q_ref.q)
        q_ref_current = q_ref

        print(f"Running attitude controller simulation "
              f"for {self.total_time}s...")

        for i in range(n_steps):
            t = i * self.dt

            # Current state with noise (Paper 2: noise=0.1)
            omega_noisy = (self.drone.omega
                          + np.random.normal(0, noise_std, 3))

            # Compute error FIRST (before control)
            q_err   = self.controller.compute_error(
                q_ref_current, self.drone.q
            )
            err_mag = np.linalg.norm(
                [q_err.q1, q_err.q2, q_err.q3]
            )

            # Advance path only once error has converged
            if err_mag < error_threshold and u_current < 0.99:
                u_current = min(u_current + u_step, 0.99)
                _, q_ref_new = self.converter.get_reference_at(
                    self.spline, u_current
                )
                q_ref_current = self.converter.slerp_static(
                    q_ref_current, q_ref_new, 0.3
                )

            # P² control law — Paper 2 Eq(21)
            tau, axis_err = self.controller.compute_torque(
                q_ref_current,
                self.drone.q,
                omega_noisy
            )

            # Step attitude dynamics only
            self.drone.step_attitude_only(tau, self.dt)

            # Log everything
            self.results['t'].append(t)
            self.results['euler'].append(
                self.drone.q.to_euler())
            self.results['euler_ref'].append(
                q_ref_current.to_euler())
            self.results['q_measured'].append(
                self.drone.q.q.copy())
            self.results['q_ref'].append(
                q_ref_current.q.copy())
            self.results['axis_error'].append(
                axis_err.copy())
            self.results['torque'].append(tau.copy())
            self.results['omega'].append(
                self.drone.omega.copy())
            self.results['u_param'].append(u_current)
            self.results['err_mag'].append(err_mag)

            # Progress
            if i % (n_steps // 10) == 0:
                pct = 100 * i // n_steps
                print(f"  {pct:3d}% | t={t:.1f}s | "
                      f"u={u_current:.3f} | "
                      f"AttErr={err_mag:.4f} | "
                      f"τ_max={np.max(np.abs(tau)):.3f}Nm")

        print("Simulation complete.")
        return self.results
