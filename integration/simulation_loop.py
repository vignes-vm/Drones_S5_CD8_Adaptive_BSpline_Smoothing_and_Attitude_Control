import numpy as np
from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from .trajectory_converter import TrajectoryConverter

class MergedSimulation:

    def __init__(self, spline, dt=0.01, total_time=30.0):
        
        self.spline = spline
        self.dt = dt
        self.total_time = total_time

        # Initialize subsystems
        self.drone = QuadrotorModel()
        self.controller = P2Controller(Pq=20.0, Pw=4.0)
        self.converter = TrajectoryConverter()

        # Generate reference sequence
        self.reference = self.converter.generate_time_sequence(
            spline, total_time, dt
        )

        # Results storage
        self.results = {
            't': [], 'position': [], 'euler': [],
            'q_measured': [], 'q_ref': [],
            'axis_error': [], 'torque': [],
            'curvature': []
        }

    def run(self, noise_std=0.01):
        
        print(f"Running merged simulation for {self.total_time}s...")
        n_steps = len(self.reference)

        # Set initial drone position to path start
        start_pos = self.spline.evaluate(0)
        self.drone.reset(position=start_pos)

        for i, ref in enumerate(self.reference):
            t        = ref['t']
            q_ref    = ref['q_ref']
            pos_ref  = ref['position']

            # Get current drone state
            q_measured = self.drone.q
            omega      = self.drone.omega

            # Add noise to measurements
            omega_noisy = omega + np.random.normal(0, noise_std, 3)

            # Compute control torque — Paper 2 Eq (21)
            tau, axis_err = self.controller.compute_torque(
                q_ref, q_measured, omega_noisy
            )

            # Step drone dynamics forward
            self.drone.step(tau, self.dt, noise_std=0)

            # Log results
            self.results['t'].append(t)
            self.results['position'].append(
                self.drone.position.copy()
            )
            self.results['euler'].append(
                self.drone.q.to_euler()
            )
            self.results['q_measured'].append(
                self.drone.q.q.copy()
            )
            self.results['q_ref'].append(q_ref.q.copy())
            self.results['axis_error'].append(axis_err.copy())
            self.results['torque'].append(tau.copy())

            # Progress update
            if i % (n_steps // 10) == 0:
                pct = 100 * i // n_steps
                print(f"  Progress: {pct}%")

        print("Simulation complete.")
        return self.results