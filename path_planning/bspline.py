import numpy as np
try:
    from scipy.interpolate import BSpline as ScipyBSpline
except ImportError:
    ScipyBSpline = None

class BSplinePath:

    def __init__(self, waypoints, knot_vector=None, degree=3,
                 param_mode='chord', n_ctrl=None):
        self.waypoints = np.array(waypoints)
        self.degree = degree
        self.n_waypoints = len(waypoints)
        self.param_mode = param_mode

        if param_mode == 'uniform':
            self.params = self._uniform_params()
        else:
            self.params = self._chord_length_params()

        if knot_vector is None:
            target_n_ctrl = n_ctrl if n_ctrl is not None else self.n_waypoints
            self.knot_vector = self._uniform_knots(target_n_ctrl)
        else:
            self.knot_vector = knot_vector

        self.control_points = self._fit_control_points()

    def _chord_length_params(self):
        diffs = np.diff(self.waypoints, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(dists)])
        total = cumulative[-1]
        if total < 1e-10:
            return np.linspace(0, 1, self.n_waypoints)
        return cumulative / total

    def _uniform_params(self):
        return np.linspace(0, 1, self.n_waypoints)

    def _uniform_knots(self, n_ctrl=None):
        n = n_ctrl if n_ctrl is not None else self.n_waypoints
        k = self.degree
        n_knots = n + k + 1
        inner = np.linspace(0, 1, n_knots - 2*(k))
        knots = np.concatenate([
            np.zeros(k),
            inner,
            np.ones(k)
        ])
        return knots

    def _fit_control_points(self):
        k = self.degree
        t = self.knot_vector
        u = self.params
        n_ctrl = len(t) - k - 1

        N = np.zeros((self.n_waypoints, n_ctrl))
        for i, ui in enumerate(u):
            ui = np.clip(ui, t[k], t[-k-1] - 1e-10)
            for j in range(n_ctrl):
                N[i, j] = self._basis(j, k, ui, t)

        NtN = N.T @ N
        NtP = N.T @ self.waypoints
        try:
            ctrl_pts = np.linalg.lstsq(NtN, NtP, rcond=None)[0]
        except:
            ctrl_pts = self.waypoints.copy()
        return ctrl_pts

    def _basis(self, i, k, u, t):
        if k == 0:
            if t[i] <= u < t[i+1]:
                return 1.0
            elif u == t[-1] and t[i] <= u <= t[i+1]:
                return 1.0
            return 0.0
        else:
            denom1 = t[i+k] - t[i]
            denom2 = t[i+k+1] - t[i+1]
            term1 = 0.0
            term2 = 0.0
            if denom1 > 1e-10:
                term1 = ((u - t[i]) / denom1) * self._basis(i, k-1, u, t)
            if denom2 > 1e-10:
                term2 = ((t[i+k+1] - u) / denom2) * self._basis(i+1, k-1, u, t)
            return term1 + term2

    def evaluate(self, u):
        u = np.clip(u, 0, 1-1e-10)
        k = self.degree
        t = self.knot_vector
        n_ctrl = len(self.control_points)
        point = np.zeros(3)
        for i in range(n_ctrl):
            b = self._basis(i, k, u, t)
            point += b * self.control_points[i]
        return point

    def evaluate_derivative(self, u, order=1):
        h = 1e-5
        u = np.clip(u, h, 1-h)
        if order == 1:
            return (self.evaluate(u+h) - self.evaluate(u-h)) / (2*h)
        elif order == 2:
            return (self.evaluate(u+h) - 2*self.evaluate(u) + self.evaluate(u-h)) / (h**2)

    def compute_curvature(self, u):
        d1 = self.evaluate_derivative(u, 1)
        d2 = self.evaluate_derivative(u, 2)
        cross = np.cross(d1, d2)
        numerator = np.linalg.norm(cross)
        denominator = np.linalg.norm(d1)**3
        if denominator < 1e-10:
            return 0.0
        return numerator / denominator

    def sample_path(self, n_points=500):
        params = np.linspace(0, 1-1e-10, n_points)
        return np.array([self.evaluate(u) for u in params])

    def path_length(self, n_points=500):
        pts = self.sample_path(n_points)
        diffs = np.diff(pts, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))

    def max_curvature(self, n_points=200):
        params = np.linspace(0.01, 0.99, n_points)
        curvatures = [self.compute_curvature(u) for u in params]
        return max(curvatures)