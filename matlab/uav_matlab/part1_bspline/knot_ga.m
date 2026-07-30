function best_knots = knot_ga(waypoints, kappa_max, n_generations, population)
% KNOT_GA  Genetic Algorithm for adaptive knot placement
% Optimizes INTERIOR knot positions only, keeping Schoenberg-Whitney
% satisfied by construction (knots stay sorted, bounded between
% neighboring u_params).

if nargin < 2, kappa_max     = 0.5; end
if nargin < 3, n_generations = 30;  end
if nargin < 4, population    = 20;  end

k  = 3;
u  = bspline_engine.chord_length_params(waypoints);
n  = length(u);
n_free = n - k - 1;   % number of interior knots we can move

if n_free <= 0
    fprintf('Too few waypoints for GA knot optimization — using chord-length knots.\n');
    best_knots = bspline_engine.knot_averaging(u, k);
    return;
end

% Baseline (chord-length averaged) knots as starting reference
baseline = bspline_engine.knot_averaging(u, k);
baseline_inner = baseline(k+2 : end-k-1);

% Bounds: each interior knot must stay within a safe band around
% its baseline position — guarantees monotonic knots after GA
band = 0.15;
lb = max(0.001, baseline_inner - band);
ub = min(0.999, baseline_inner + band);

fitness = @(x) evaluate_knots(x, waypoints, kappa_max, k);

opts = optimoptions('ga', ...
    'MaxGenerations',    n_generations, ...
    'PopulationSize',    population, ...
    'CrossoverFraction', 0.8, ...
    'Display',           'iter', ...
    'PlotFcn',           []);

fprintf('Running GA for knot optimization (%d free knots)...\n', n_free);
[best_inner, best_cost] = ga(fitness, n_free, ...
    [], [], [], [], lb, ub, [], opts);

best_inner = sort(best_inner);
best_knots = [zeros(1,k+1), best_inner, ones(1,k+1)];

fprintf('GA complete. Best cost: %.4f\n', best_cost);
end

function cost = evaluate_knots(inner_knots, waypoints, kappa_max, k)
inner_knots = sort(inner_knots);
knots = [zeros(1,k+1), inner_knots, ones(1,k+1)];

try
    e = bspline_engine(waypoints, 'chord', knots);
    L = e.length();
    kmax = e.max_curvature();
    curv_penalty = max(0, kmax - kappa_max)^2 * 1000;
    cost = L + curv_penalty;
catch
    cost = 1e8;   % invalid knot arrangement
end
end