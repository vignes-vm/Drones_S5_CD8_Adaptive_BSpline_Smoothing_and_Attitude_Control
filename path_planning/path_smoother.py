"""
Top-level orchestrator for Paper 1's full three-stage adaptive B-Spline
smoothing algorithm (§3.4 Algorithmic Flow):

    1. Initialization      — least-squares control polygon + baseline knots
    2. Global Optimization — GA search over the knot vector (knot_ga.py)
    3. Local Refinement    — curvature-aware control-point adjustment
                              (curvature_refine.py)

Ties BSplinePath, KnotGA, and CurvatureRefiner together behind a single
call so scripts don't have to hand-wire all three stages themselves.
"""
import numpy as np

from .bspline import BSplinePath
from .knot_ga import KnotGA
from .curvature_refine import CurvatureRefiner


class AdaptiveBSplineSmoother:

    def __init__(self, kappa_max=0.5, n_inner_knots=8,
                 n_generations=50, population_size=30,
                 refine_max_iters=30, refine_step_size=5.0):
        self.kappa_max = kappa_max
        self.n_inner_knots = n_inner_knots
        self.n_generations = n_generations
        self.population_size = population_size
        self.refine_max_iters = refine_max_iters
        self.refine_step_size = refine_step_size

        # Populated by smooth() — one spline per stage, so callers can
        # compare before/after each step (e.g. for the recreation plots).
        self.stage1_spline = None   # baseline (uniform knots)
        self.stage2_spline = None   # after GA knot optimization
        self.stage3_spline = None   # after curvature refinement (final)
        self.ga = None
        self.refiner = None

    def smooth(self, waypoints, verbose=True):
        waypoints = np.asarray(waypoints, dtype=float)

        # ── Stage 1 — Initialization ──
        self.stage1_spline = BSplinePath(waypoints)
        if verbose:
            print(f"[Stage 1] Baseline spline:      "
                  f"length={self.stage1_spline.path_length():.1f}m, "
                  f"max_curvature={self.stage1_spline.max_curvature():.4f}")

        # ── Stage 2 — Global Optimization (GA over knots) ──
        self.ga = KnotGA(
            waypoints,
            n_inner_knots=self.n_inner_knots,
            n_generations=self.n_generations,
            population_size=self.population_size,
            kappa_max=self.kappa_max
        )
        best_knots = self.ga.run()
        self.stage2_spline = BSplinePath(waypoints, best_knots)
        if verbose:
            print(f"[Stage 2] GA-optimized spline:   "
                  f"length={self.stage2_spline.path_length():.1f}m, "
                  f"max_curvature={self.stage2_spline.max_curvature():.4f}")

        # ── Stage 3 — Local Curvature Refinement ──
        # Refit a fresh spline on the same (waypoints, knots) rather than
        # reusing stage2_spline, so stage2_spline is left untouched for
        # before/after comparison.
        self.stage3_spline = BSplinePath(waypoints, best_knots)
        self.refiner = CurvatureRefiner(
            kappa_max=self.kappa_max,
            max_iters=self.refine_max_iters,
            step_size=self.refine_step_size
        )
        self.refiner.refine(self.stage3_spline, verbose=verbose)
        if verbose:
            print(f"[Stage 3] Curvature-refined spline: "
                  f"length={self.stage3_spline.path_length():.1f}m, "
                  f"max_curvature={self.stage3_spline.max_curvature():.4f}")

        return self.stage3_spline
