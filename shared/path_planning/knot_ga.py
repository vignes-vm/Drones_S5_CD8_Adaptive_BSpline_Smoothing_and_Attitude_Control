import numpy as np
try:
    import pygad
    PYGAD_AVAILABLE = True
except ImportError:
    PYGAD_AVAILABLE = False
    pygad = None


class KnotGA:

    def __init__(self, waypoints, degree=3,
                 n_inner_knots=8,
                 n_generations=50,
                 population_size=30,
                 kappa_max=0.5):

        self.waypoints = waypoints
        self.degree = degree
        self.n_inner = n_inner_knots
        self.n_generations = n_generations
        self.pop_size = population_size
        self.kappa_max = kappa_max

        # Weights for objective function
        self.w1 = 1.0   # data fidelity
        self.w2 = 2.0   # curvature penalty
        self.w3 = 1.0   # path length
        self.w4 = 0.5   # knot regularization

        self.best_knots = None
        self.best_fitness = -np.inf

    def build_knot_vector(self, inner_knots):
        inner = np.sort(np.clip(inner_knots, 1e-4, 1-1e-4))
        k = self.degree
        knots = np.concatenate([
            np.zeros(k+1),
            inner,
            np.ones(k+1)
        ])
        return knots

    def evaluate_chromosome(self, inner_knots):
        from .bspline import BSplinePath

        knots = self.build_knot_vector(inner_knots)

        try:
            spline = BSplinePath(self.waypoints, knots)
        except:
            return 1e6   # invalid chromosome

        # Term 1 — Data fidelity (mean squared error)
        fidelity = 0
        for i, wp in enumerate(self.waypoints):
            u = spline.params[i]
            pt = spline.evaluate(u)
            fidelity += np.linalg.norm(wp - pt)**2
        fidelity /= len(self.waypoints)

        # Term 2 — Curvature penalty
        params = np.linspace(0.01, 0.99, 50)
        curvatures = [spline.compute_curvature(u) for u in params]
        curv_penalty = sum(max(0, k - self.kappa_max)**2
                          for k in curvatures)

        # Term 3 — Path length
        path_len = spline.path_length(200)

        # Term 4 — Knot regularization
        diffs = np.diff(inner_knots)
        avg_diff = np.mean(diffs) if len(diffs) > 0 else 0
        knot_reg = np.sum((diffs - avg_diff)**2)

        total_cost = (self.w1 * fidelity +
                      self.w2 * curv_penalty +
                      self.w3 * path_len +
                      self.w4 * knot_reg)

        return total_cost

    def fitness_function(self, ga_instance, solution, sol_idx):
        cost = self.evaluate_chromosome(solution)
        return -cost

    def _run_numpy_ga(self):
        rng = np.random.default_rng(42)
        pop = rng.uniform(0.01, 0.99, size=(self.pop_size, self.n_inner))
        best_sol = None
        best_cost = float('inf')

        for gen in range(self.n_generations):
            costs = np.array([self.evaluate_chromosome(ind) for ind in pop])
            min_idx = np.argmin(costs)
            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_sol = pop[min_idx].copy()

            # Tournament selection & crossover
            new_pop = [best_sol.copy(), pop[min_idx].copy()]
            while len(new_pop) < self.pop_size:
                idx1, idx2 = rng.choice(self.pop_size, 2, replace=False)
                p1 = pop[idx1] if costs[idx1] < costs[idx2] else pop[idx2]
                idx3, idx4 = rng.choice(self.pop_size, 2, replace=False)
                p2 = pop[idx3] if costs[idx3] < costs[idx4] else pop[idx4]

                xpoint = rng.integers(1, self.n_inner)
                child = np.concatenate([p1[:xpoint], p2[xpoint:]])

                # Mutation
                if rng.random() < 0.3:
                    m_idx = rng.integers(0, self.n_inner)
                    child[m_idx] = rng.uniform(0.01, 0.99)

                new_pop.append(child)

            pop = np.array(new_pop[:self.pop_size])

        self.best_knots = self.build_knot_vector(np.sort(best_sol))
        self.best_fitness = -best_cost
        print(f"GA complete (NumPy fallback). Best cost: {best_cost:.4f}", flush=True)
        return self.best_knots

    def run(self):
        print("Running Genetic Algorithm for knot optimization...", flush=True)

        if not PYGAD_AVAILABLE:
            return self._run_numpy_ga()

        try:
            ga = pygad.GA(
                num_generations=self.n_generations,
                num_parents_mating=self.pop_size // 3,
                fitness_func=self.fitness_function,
                sol_per_pop=self.pop_size,
                num_genes=self.n_inner,
                gene_space={'low': 0.01, 'high': 0.99},
                mutation_percent_genes=20,
                crossover_type="single_point",
                mutation_type="random",
                keep_parents=2,
                suppress_warnings=True
            )
            ga.run()

            best_solution, best_fitness, _ = ga.best_solution()
            self.best_knots = self.build_knot_vector(np.sort(best_solution))
            self.best_fitness = best_fitness
            print(f"GA complete. Best cost: {-best_fitness:.4f}", flush=True)
            return self.best_knots
        except Exception:
            return self._run_numpy_ga()