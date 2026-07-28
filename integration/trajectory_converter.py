import numpy as np
from attitude_control.quaternion import Quaternion

class TrajectoryConverter:

    def path_tangent_to_quaternion(self, tangent, up=None):
        if up is None:
            up = np.array([0, 0, 1])

        # Normalize tangent
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            return Quaternion()   # identity

        x_body = tangent / tangent_norm

        # Handle case where tangent is parallel to up
        if abs(np.dot(x_body, up)) > 0.99:
            up = np.array([0, 1, 0])

        # Build orthonormal frame
        y_body = np.cross(up, x_body)
        y_body = y_body / np.linalg.norm(y_body)
        z_body = np.cross(x_body, y_body)
        z_body = z_body / np.linalg.norm(z_body)

        # Rotation matrix
        R = np.column_stack([x_body, y_body, z_body])

        # Convert to quaternion
        return Quaternion.from_rotation_matrix(R)

    def get_reference_at(self, spline, u):
        position = spline.evaluate(u)
        tangent  = spline.evaluate_derivative(u, 1)
        q_ref    = self.path_tangent_to_quaternion(tangent)
        return position, q_ref

    def generate_time_sequence(self, spline,
                               total_time=30.0,
                               dt=0.01):

        n_steps = int(total_time / dt)
        params = np.linspace(0, 0.99, n_steps)

        sequence = []
        for i, u in enumerate(params):
            t = i * dt
            pos, q_ref = self.get_reference_at(spline, u)
            sequence.append({
                't': t,
                'u': u,
                'position': pos,
                'q_ref': q_ref
            })
        return sequence