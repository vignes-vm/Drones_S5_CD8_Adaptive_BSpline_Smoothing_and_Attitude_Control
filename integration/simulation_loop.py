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
    This directly mirrors Paper 2's simulation methodology
    but with path-derived references instead of manual ones.
    """

    def __init__(self, spline, dt=0.01, total_time=20.0,
                 Pq=20.0, Pw=4.0):
        self.spline      = spline
        self.dt          = dt
        self.total_time  = total_time

        # Paper 2 gains exactly as specified
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
            'u_param':    []
        }

    def generate_attitude_sequence(self):
        """
        Sample the B-Spline path and extract
        desired attitude at each time step.
        This is the bridge between Paper 1 and Paper 2.
        """
        n_steps = int(self.total_time / self.dt)
        # Map time linearly to path parameter
        u_values = np.linspace(0.001, 0.98, n_steps)
        sequence = []
        for u in u_values:
            _, q_ref = self.converter.get_reference_at(
                self.spline, u
            )
            sequence.append((u, q_ref))
        return sequence

    def run(self, noise_std=0.1):
        """
        Run attitude tracking simulation.
        noise_std=0.1 matches Paper 2 exactly.
        """
        print(f"Generating attitude reference sequence "
              f"from B-Spline path...")
        sequence = self.generate_attitude_sequence()
        n_steps  = len(sequence)

        print(f"Running attitude controller simulation "
              f"for {self.total_time}s...")

        # Initialize drone — start aligned with path start
        _, q_init = self.converter.get_reference_at(
            self.spline, 0.001
        )
        self.drone.reset()
        self.drone.q = q_init

        # Smooth reference tracker
        q_ref_smooth = q_init

        for i, (u, q_ref) in enumerate(sequence):
            t = i * self.dt

            # Smooth reference — prevents controller
            # chasing instantaneous reference changes
            alpha        = 0.04
            q_ref_smooth = self.converter.slerp_static(
                q_ref_smooth, q_ref, alpha
            )

            # Current state with noise (Paper 2: noise=0.1)
            omega_noisy = (self.drone.omega
                          + np.random.normal(0, noise_std, 3))

            # P² control law — Paper 2 Eq(21)
            tau, axis_err = self.controller.compute_torque(
                q_ref_smooth,
                self.drone.q,
                omega_noisy
            )

            # Step attitude dynamics only
            # (Paper 2 does this — attitude simulation)
            self.drone.step_attitude_only(tau, self.dt)

            # Log everything
            self.results['t'].append(t)
            self.results['euler'].append(
                self.drone.q.to_euler())
            self.results['euler_ref'].append(
                q_ref_smooth.to_euler())
            self.results['q_measured'].append(
                self.drone.q.q.copy())
            self.results['q_ref'].append(
                q_ref_smooth.q.copy())
            self.results['axis_error'].append(
                axis_err.copy())
            self.results['torque'].append(tau.copy())
            self.results['omega'].append(
                self.drone.omega.copy())
            self.results['u_param'].append(u)

            # Progress
            if i % (n_steps // 10) == 0:
                pct     = 100 * i // n_steps
                err_mag = np.linalg.norm(axis_err)
                print(f"  {pct:3d}% | t={t:.1f}s | "
                      f"u={u:.3f} | "
                      f"AttErr={err_mag:.4f} | "
                      f"τ_max={np.max(np.abs(tau)):.3f}Nm")

        print("Simulation complete.")
        return self.results