import numpy as np
from attitude_control.quaternion import Quaternion

class TrajectoryConverter:

    def __init__(self):
        self.prev_q_ref = None   # for smoothing

    def path_tangent_to_quaternion(self, tangent, up=None):

        if up is None:
            up = np.array([0.0, 0.0, 1.0])

        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-6:
            return self.prev_q_ref if self.prev_q_ref else Quaternion()

        # Drone x-axis points along path
        x_body = tangent / tangent_norm

        # Avoid singularity when tangent parallel to up
        if abs(np.dot(x_body, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])

        # Orthonormal frame
        y_body = np.cross(up, x_body)
        y_norm = np.linalg.norm(y_body)
        if y_norm < 1e-6:
            return self.prev_q_ref if self.prev_q_ref else Quaternion()
        y_body = y_body / y_norm

        z_body = np.cross(x_body, y_body)
        z_body = z_body / np.linalg.norm(z_body)

        # Build rotation matrix — columns are body axes in world frame
        R = np.column_stack([x_body, y_body, z_body])

        # Ensure valid rotation matrix
        if abs(np.linalg.det(R) - 1.0) > 0.01:
            return self.prev_q_ref if self.prev_q_ref else Quaternion()

        q_new = Quaternion.from_rotation_matrix(R)

        # Ensure quaternion continuity — flip if needed to take
        # shortest path from previous reference
        if self.prev_q_ref is not None:
            dot = np.dot(q_new.q, self.prev_q_ref.q)
            if dot < 0:
                q_new = Quaternion(-q_new.q0, -q_new.q1,
                                   -q_new.q2, -q_new.q3)

        self.prev_q_ref = q_new
        return q_new

    def get_reference_at(self, spline, u):

        position = spline.evaluate(u)
        tangent  = spline.evaluate_derivative(u, 1)

        # Smooth tangent — average with small window
        h = 0.005
        t1 = spline.evaluate_derivative(max(0.001, u - h), 1)
        t2 = spline.evaluate_derivative(min(0.999, u + h), 1)
        tangent_smooth = (t1 + tangent + t2) / 3.0

        q_ref = self.path_tangent_to_quaternion(tangent_smooth)
        return position, q_ref

    def to_euler(self, q):
        return q.to_euler()

    def generate_time_sequence(self, spline,
                               total_time=20.0,
                               dt=0.01):

        n_steps = int(total_time / dt)
        # Use slightly compressed range to avoid endpoint issues
        params = np.linspace(0.001, 0.98, n_steps)

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
    
    @staticmethod
    def slerp_static(q1, q2, t):
        """
        Static slerp — no instance needed.
        Smooth interpolation between two quaternions.
        """
        dot = np.dot(q1.q, q2.q)

        if dot < 0:
            q2  = Quaternion(-q2.q0, -q2.q1,
                             -q2.q2, -q2.q3)
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.9995:
            result = q1.q + t * (q2.q - q1.q)
            norm   = np.linalg.norm(result)
            return Quaternion(*(result / norm))

        theta_0    = np.arccos(dot)
        theta      = theta_0 * t
        sin_t      = np.sin(theta)
        sin_t0     = np.sin(theta_0)

        s1 = np.cos(theta) - dot * sin_t / sin_t0
        s2 = sin_t / sin_t0

        result = s1 * q1.q + s2 * q2.q
        return Quaternion(*result)