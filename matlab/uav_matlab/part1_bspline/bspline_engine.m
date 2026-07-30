classdef bspline_engine
% BSPLINE_ENGINE  Self-contained cubic B-Spline interpolation
% Implements Piegl & Tiller "Global Curve Interpolation" (NURBS Book,
% Algorithm A9.1) with knot averaging — proven well-conditioned by
% de Boor (1978).
%
% USAGE:
%   e = bspline_engine(waypoints, 'uniform');    % uniform u_params
%   e = bspline_engine(waypoints, 'chord');      % chord-length u_params
%   e = bspline_engine(waypoints, 'chord', knots); % custom knots (GA)
%   pts = e.sample(500);
%   L   = e.length();
%   k   = e.max_curvature();

    properties
        waypoints
        u_params
        knot_vec
        ctrl_pts
        degree = 3
        n
    end

    methods
        function obj = bspline_engine(waypoints, param_method, custom_knots)
            if nargin < 2, param_method = 'chord'; end
            if nargin < 3, custom_knots = []; end

            obj.waypoints = waypoints;
            obj.n         = size(waypoints, 1);
            k             = obj.degree;

            % ── Parameter values u_i ──
            switch param_method
                case 'uniform'
                    obj.u_params = linspace(0, 1, obj.n);
                case 'chord'
                    obj.u_params = obj.chord_length_params(waypoints);
                otherwise
                    error('param_method must be ''uniform'' or ''chord''');
            end

            % ── Knot vector ──
            if isempty(custom_knots)
                obj.knot_vec = obj.knot_averaging(obj.u_params, k);
            else
                obj.knot_vec = custom_knots(:)';
            end

            n_ctrl_expected = length(obj.knot_vec) - k - 1;
            if n_ctrl_expected ~= obj.n
                error(['Knot vector produces %d control points but ' ...
                       'have %d waypoints. They must match for ' ...
                       'interpolation.'], n_ctrl_expected, obj.n);
            end

            % ── Solve for control points ──
            B = obj.basis_matrix(obj.u_params, obj.knot_vec, k, obj.n);

            cn = cond(B);
            if cn > 1e8
                warning(['bspline_engine: basis matrix condition ' ...
                    'number = %.2e (>1e8). Fit may be unstable. ' ...
                    'This usually means u_params and knot_vec are ' ...
                    'mismatched (Schoenberg-Whitney violated).'], cn);
            end

            obj.ctrl_pts = B \ waypoints;

            % ── Self-test: does the curve actually pass near waypoints? ──
            obj.self_test();
        end

        function self_test(obj)
        % Verify interpolation actually worked — catch bugs immediately
            max_err = 0;
            for i = 1:obj.n
                pt = obj.evaluate(obj.u_params(i));
                err = norm(pt - obj.waypoints(i,:));
                max_err = max(max_err, err);
            end
            if max_err > 1.0   % more than 1 meter off is a red flag
                error(['SELF-TEST FAILED: B-Spline does not pass ' ...
                    'through waypoints (max error = %.2f m). ' ...
                    'Fit is broken — check knot vector.'], max_err);
            end
            fprintf('  [self-test OK: max fit error = %.2e m]\n', max_err);
        end

        function pt = evaluate(obj, u)
            k  = obj.degree;
            u  = max(obj.knot_vec(k+1), ...
                     min(obj.knot_vec(end-k) - 1e-12, u));
            pt = zeros(1,3);
            for j = 1:obj.n
                b  = obj.deBoor(j-1, k, u, obj.knot_vec);
                pt = pt + b * obj.ctrl_pts(j,:);
            end
        end

        function pts = sample(obj, n_samples)
            if nargin < 2, n_samples = 500; end
            k    = obj.degree;
            u_lo = obj.knot_vec(k+1);
            u_hi = obj.knot_vec(end-k);
            u_ev = linspace(u_lo, u_hi, n_samples);
            pts  = zeros(n_samples, 3);
            for i = 1:n_samples
                pts(i,:) = obj.evaluate(u_ev(i));
            end
        end

        function L = length(obj)
            pts = obj.sample(500);
            d   = diff(pts);
            L   = sum(sqrt(sum(d.^2, 2)));
        end

        function kmax = max_curvature(obj)
            k      = obj.degree;
            u_lo   = obj.knot_vec(k+1);
            u_hi   = obj.knot_vec(end-k) - 1e-10;
            u_vals = linspace(u_lo+0.01, u_hi-0.01, 200);
            h      = 1e-5;
            kappa  = zeros(size(u_vals));
            for i = 1:length(u_vals)
                u  = u_vals(i);
                C0 = obj.evaluate(u);
                Cp = obj.evaluate(min(u+h, u_hi));
                Cm = obj.evaluate(max(u-h, u_lo));
                d1 = (Cp - Cm) / (2*h);
                d2 = (Cp - 2*C0 + Cm) / h^2;
                nd = norm(d1);
                if nd > 1e-10
                    kappa(i) = norm(cross(d1,d2)) / nd^3;
                end
            end
            kmax = max(kappa);
        end

        function tangent = tangent_at(obj, u)
            h  = 1e-5;
            k  = obj.degree;
            u_lo = obj.knot_vec(k+1);
            u_hi = obj.knot_vec(end-k) - 1e-10;
            Cp = obj.evaluate(min(u+h, u_hi));
            Cm = obj.evaluate(max(u-h, u_lo));
            tangent = (Cp - Cm) / (2*h);
        end
    end

    methods (Static)
        function u = chord_length_params(waypoints)
            d = diff(waypoints);
            dist = sqrt(sum(d.^2, 2));
            cum  = [0; cumsum(dist)];
            tot  = cum(end);
            if tot < 1e-10
                u = linspace(0,1,size(waypoints,1));
            else
                u = (cum / tot)';
            end
        end

        function knots = knot_averaging(u, p)
        % Piegl & Tiller Eq. 9.8 — de Boor 1978 knot averaging
        % Guarantees Schoenberg-Whitney condition => well-conditioned
            u = u(:)';
            n = length(u);          % n data points, n control points
            knots = zeros(1, n+p+1);
            knots(1:p+1)     = 0;
            knots(end-p:end) = 1;
            for j = 1:(n-p-1)
                knots(j+p+1) = mean(u(j+1 : j+p));
            end
        end

        function B = basis_matrix(u_params, knot_vec, k, n_ctrl)
            N = length(u_params);
            B = zeros(N, n_ctrl);
            for i = 1:N
                u = u_params(i);
                u = max(knot_vec(k+1), min(knot_vec(end-k)-1e-12, u));
                for j = 1:n_ctrl
                    B(i,j) = bspline_engine.deBoor(j-1, k, u, knot_vec);
                end
            end
        end

        function val = deBoor(i, k, u, t)
            if k == 0
                lo = t(i+1); hi = t(i+2);
                if lo <= u && u < hi
                    val = 1;
                elseif u >= t(end) - 1e-12 && u <= hi
                    val = 1;
                else
                    val = 0;
                end
                return
            end
            d1 = t(i+k+1) - t(i+1);
            d2 = t(i+k+2) - t(i+2);
            t1 = 0; t2 = 0;
            if d1 > 1e-12
                t1 = ((u - t(i+1)) / d1) * bspline_engine.deBoor(i, k-1, u, t);
            end
            if d2 > 1e-12
                t2 = ((t(i+k+2) - u) / d2) * bspline_engine.deBoor(i+1, k-1, u, t);
            end
            val = t1 + t2;
        end
    end
end