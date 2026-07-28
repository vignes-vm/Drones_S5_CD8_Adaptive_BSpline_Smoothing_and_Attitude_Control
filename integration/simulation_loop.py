import numpy as np
from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from .trajectory_converter import TrajectoryConverter
from attitude_control.quaternion import Quaternion

class MergedSimulation:

    def __init__(self, spline, dt=0.01, total_time=20.0,
                 Pq=0.05, Pw=0.01):
        self.spline = spline
        self.dt = dt
        self.total_time = total_time

        self.drone    = QuadrotorModel()
        self.controller = P2Controller(Pq=Pq, Pw=Pw)
        self.converter  = TrajectoryConverter()

        self.reference = self.converter.generate_time_sequence(
            spline, total_time, dt
        )

        self.results = {
            't': [], 'position': [], 'euler': [],
            'q_measured': [], 'q_ref': [],
            'axis_error': [], 'torque': [],
            'omega': []
        }

    def slerp(self, q1, q2, t):
        
        dot = np.dot(q1.q, q2.q)

        # Ensure shortest path
        if dot < 0:
            q2 = Quaternion(-q2.q0, -q2.q1, -q2.q2, -q2.q3)
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.9995:
            # Very close — linear interpolation
            result = q1.q + t * (q2.q - q1.q)
            return Quaternion(*result)

        theta_0 = np.arccos(dot)
        theta   = theta_0 * t
        sin_theta   = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        result = s1 * q1.q + s2 * q2.q
        return Quaternion(*result)

    def run(self, noise_std=0.01, ref_smoothing=0.05):

        print(f"Running merged simulation for {self.total_time}s...")
        n_steps = len(self.reference)

        # Initialize drone at path start
        start_pos = self.spline.evaluate(0.001)
        self.drone.reset(position=start_pos)

        # Start with first reference as current smoothed reference
        q_ref_smooth = self.reference[0]['q_ref']

        for i, ref in enumerate(self.reference):
            t       = ref['t']
            q_ref   = ref['q_ref']

            # Smooth the reference quaternion over time
            # Prevents controller from chasing rapidly changing targets
            q_ref_smooth = self.slerp(q_ref_smooth, q_ref,
                                      ref_smoothing)

            # Current drone state
            q_measured = self.drone.q
            omega      = self.drone.omega

            # Add sensor noise
            omega_noisy = omega + np.random.normal(0, noise_std, 3)

            # Compute control — Paper 2 Eq (21)
            tau, axis_err = self.controller.compute_torque(
                q_ref_smooth, q_measured, omega_noisy
            )

            # Step dynamics
            self.drone.step(tau, self.dt)

            # Log
            self.results['t'].append(t)
            self.results['euler'].append(
                self.drone.q.to_euler()
            )
            self.results['q_measured'].append(
                self.drone.q.q.copy()
            )
            self.results['q_ref'].append(q_ref_smooth.q.copy())
            self.results['axis_error'].append(axis_err.copy())
            self.results['torque'].append(tau.copy())
            self.results['omega'].append(omega.copy())

            if i % (n_steps // 10) == 0:
                pct = 100 * i // n_steps
                err_mag = np.linalg.norm(axis_err)
                print(f"  {pct}% | Error magnitude: {err_mag:.4f} | "
                      f"Max torque: {np.max(np.abs(tau)):.3f} Nm")

        print("Simulation complete.")
        return self.results