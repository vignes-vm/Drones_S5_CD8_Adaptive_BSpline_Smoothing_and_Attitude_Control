"""
Paper 1 §3.3 — Curvature-aware Control Point Adjustment.

This is stage 3 of the paper's algorithmic flow (§3.4):

    1. Initialization      — least-squares control polygon + baseline knots
    2. Global Optimization — GA search over the knot vector (knot_ga.py)
    3. Local Refinement    — THIS MODULE. After the GA has settled on a
                              knot vector, locally nudge individual control
                              points wherever the resulting spline's
                              curvature still exceeds kappa_max, using a
                              derivative-free local search that reduces
                              kappa(u) = ||C'(u) x C''(u)|| / ||C'(u)||^3
                              (Eq. 2) in the offending region — without
                              touching the knot vector itself.

The GA's fitness function already penalizes curvature violations when
choosing knots, but it never directly moves control points. This module
is the missing complementary step the paper specifies: a targeted,
worst-violation-first local search over control-point positions.
"""
import numpy as np


class CurvatureRefiner:

    def __init__(self, kappa_max=0.5, max_iters=30, step_size=5.0,
                 n_samples=200, shrink=0.5, min_step=0.05,
                 finite_diff_h=1e-3):
        self.kappa_max = kappa_max
        self.max_iters = max_iters
        self.step_size = step_size
        self.n_samples = n_samples
        self.shrink = shrink
        self.min_step = min_step
        self.h = finite_diff_h

    def _influencing_control_points(self, spline, u):
        """Indices of control points whose basis function has nonzero
        support at u (degree-k basis i is supported on [t[i], t[i+k+1]))."""
        k = spline.degree
        t = spline.knot_vector
        n_ctrl = len(spline.control_points)
        return [i for i in range(n_ctrl) if t[i] <= u < t[i + k + 1]]

    def refine(self, spline, verbose=False):
        """
        Mutates spline.control_points in place, iteratively targeting the
        single worst curvature violation each pass and nudging its
        influencing control points against the finite-difference gradient
        of curvature at that parameter — a derivative-free local search,
        as described in the paper. Returns the same (mutated) spline.
        """
        step = self.step_size
        converged_iter = None

        for iteration in range(self.max_iters):
            params = np.linspace(0.01, 0.99, self.n_samples)
            curvatures = np.array([spline.compute_curvature(u) for u in params])
            worst_idx = int(np.argmax(curvatures))
            worst_kappa = curvatures[worst_idx]

            if worst_kappa <= self.kappa_max:
                converged_iter = iteration
                break

            u_worst = params[worst_idx]
            influencing = self._influencing_control_points(spline, u_worst)
            if not influencing:
                break

            improved = False
            for ci in influencing:
                original = spline.control_points[ci].copy()
                grad = np.zeros(3)

                for axis in range(3):
                    plus = original.copy(); plus[axis] += self.h
                    minus = original.copy(); minus[axis] -= self.h

                    spline.control_points[ci] = plus
                    k_plus = spline.compute_curvature(u_worst)
                    spline.control_points[ci] = minus
                    k_minus = spline.compute_curvature(u_worst)
                    spline.control_points[ci] = original

                    grad[axis] = (k_plus - k_minus) / (2 * self.h)

                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-9:
                    continue

                direction = -grad / grad_norm
                spline.control_points[ci] = original + step * direction
                new_kappa = spline.compute_curvature(u_worst)

                if new_kappa < worst_kappa:
                    improved = True
                    worst_kappa = new_kappa
                else:
                    spline.control_points[ci] = original  # revert, no gain

            if verbose:
                print(f"  [refine] iter {iteration}: worst kappa "
                      f"{worst_kappa:.4f} at u={u_worst:.3f}, step={step:.3f}")

            if not improved:
                step *= self.shrink
                if step < self.min_step:
                    break

        if verbose and converged_iter is not None:
            print(f"  [refine] converged after {converged_iter} iterations "
                  f"(max curvature <= kappa_max={self.kappa_max})")
        elif verbose:
            print(f"  [refine] stopped after {self.max_iters} iterations "
                  f"without full convergence")

        return spline
