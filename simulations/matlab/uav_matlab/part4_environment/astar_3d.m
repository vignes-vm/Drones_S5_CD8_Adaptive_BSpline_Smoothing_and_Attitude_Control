function [waypoints, raw_path, n_expanded] = astar_3d(env, resolution)
% ASTAR_3D  3D grid A* search avoiding obstacles in env
% Returns:
%   waypoints  - pruned sparse waypoints (for B-Spline fitting)
%   raw_path   - full dense A* path (the "baseline polyline")
%   n_expanded - number of nodes expanded (for reporting)

    if nargin < 2, resolution = 50; end

    nx = round(env.space_x/resolution) + 1;
    ny = round(env.space_y/resolution) + 1;
    nz = round(env.space_z/resolution) + 1;

    to_grid = @(p) round(p/resolution) + 1;
    to_world = @(g) (g-1)*resolution;

    start_g = to_grid(env.start_pt);
    goal_g  = to_grid(env.goal_pt);

    start_idx = sub2ind_safe([nx ny nz], start_g);
    goal_idx  = sub2ind_safe([nx ny nz], goal_g);

    % 26-connected neighbor offsets
    [dx,dy,dz] = ndgrid(-1:1,-1:1,-1:1);
    offsets = [dx(:) dy(:) dz(:)];
    offsets(all(offsets==0,2),:) = [];

    openMap  = containers.Map('KeyType','double','ValueType','double'); % gScore
    cameFrom = containers.Map('KeyType','double','ValueType','double');
    closed   = containers.Map('KeyType','double','ValueType','logical');

    openMap(start_idx) = 0;
    openSet = start_idx;
    fScore  = containers.Map('KeyType','double','ValueType','double');
    fScore(start_idx) = heuristic(start_g, goal_g, resolution);

    n_expanded = 0;
    max_iter = nx*ny*nz;

    fprintf('Running 3D A* (grid %dx%dx%d = %d cells)...\n', nx,ny,nz,nx*ny*nz);

    found = false;
    for iter = 1:max_iter
        if isempty(openSet), break; end

        % Pick node in openSet with lowest fScore
        fvals = zeros(size(openSet));
        for k = 1:numel(openSet)
            fvals(k) = fScore(openSet(k));
        end
        [~, mi] = min(fvals);
        current = openSet(mi);
        openSet(mi) = [];

        if current == goal_idx
            found = true; break;
        end

        closed(current) = true;
        n_expanded = n_expanded + 1;
        cur_g = ind2sub_safe([nx ny nz], current);

        for k = 1:size(offsets,1)
            ng = cur_g + offsets(k,:);
            if any(ng < 1) || ng(1)>nx || ng(2)>ny || ng(3)>nz
                continue;
            end
            n_idx = sub2ind_safe([nx ny nz], ng);
            if isKey(closed, n_idx), continue; end

            world_p = to_world(ng);
            if env.is_collision(world_p), continue; end

            step_cost = norm(offsets(k,:)) * resolution;
            tentative_g = openMap(current) + step_cost;

            if ~isKey(openMap, n_idx) || tentative_g < openMap(n_idx)
                cameFrom(n_idx) = current;
                openMap(n_idx)  = tentative_g;
                fScore(n_idx)   = tentative_g + heuristic(ng, goal_g, resolution);
                if ~ismember(n_idx, openSet)
                    openSet(end+1) = n_idx;
                end
            end
        end
    end

    if ~found
        error('A* failed to find a path — check obstacle density/buffer vs space size.');
    end

    fprintf('A* found path. Nodes expanded: %d\n', n_expanded);

    % Reconstruct path
    path_idx = goal_idx;
    raw_grid = ind2sub_safe([nx ny nz], goal_idx);
    node = goal_idx;
    while node ~= start_idx
        node = cameFrom(node);
        raw_grid = [ind2sub_safe([nx ny nz], node); raw_grid]; %#ok<AGROW>
    end

    raw_path = to_world(raw_grid);
    raw_path(1,:)   = env.start_pt;
    raw_path(end,:) = env.goal_pt;

    waypoints = douglas_peucker_3d(raw_path, resolution*0.6);

    fprintf('Raw A* path: %d points -> pruned to %d waypoints\n', ...
        size(raw_path,1), size(waypoints,1));
end

function h = heuristic(g1, g2, res)
    h = norm((g1-g2)*res);
end

function idx = sub2ind_safe(sz, g)
    idx = sub2ind(sz, g(1), g(2), g(3));
end

function g = ind2sub_safe(sz, idx)
    [a,b,c] = ind2sub(sz, idx);
    g = [a b c];
end

function simplified = douglas_peucker_3d(pts, epsilon)
% Douglas-Peucker simplification for 3D polyline
    if size(pts,1) < 3
        simplified = pts; return;
    end
    d_max = 0; index = 0;
    p1 = pts(1,:); p2 = pts(end,:);
    for i = 2:size(pts,1)-1
        d = point_line_dist(pts(i,:), p1, p2);
        if d > d_max
            d_max = d; index = i;
        end
    end
    if d_max > epsilon
        left  = douglas_peucker_3d(pts(1:index,:), epsilon);
        right = douglas_peucker_3d(pts(index:end,:), epsilon);
        simplified = [left(1:end-1,:); right];
    else
        simplified = [p1; p2];
    end
end

function d = point_line_dist(p, a, b)
    ab = b - a; ap = p - a;
    t = dot(ap,ab) / max(dot(ab,ab), 1e-10);
    t = max(0, min(1, t));
    proj = a + t*ab;
    d = norm(p - proj);
end