%% RUN_PART4_DENSE  Denser obstacle field forcing real path detours
% Obstacles are placed directly in the start-goal corridor so A*
% must navigate around/between them, not just skirt past at a distance.

clear; clc; close all;
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

fprintf('=== Part 4 (Dense): Obstacle-Aware Pipeline ===\n\n');

%% Step 1 — Build environment with obstacles ON the direct path
start_pt = [0, 0, 20];
goal_pt  = [2000, 1800, 200];

% Direct line from start to goal for reference:
% direction vector, used to place obstacles ON this line
dir_vec = (goal_pt - start_pt) / norm(goal_pt - start_pt);

env = obstacle_env(start_pt, goal_pt);

% Place obstacles at fractions t=0.15..0.85 along the direct path,
% with small lateral offsets so A* must weave between them
fractions = [0.15, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88];
lateral_offsets = [80, -100, 60, -80, 100, -60, 40];   % meters, alternating sides
types = {'cuboid','cylinder','cuboid','cylinder','cuboid','cylinder','cuboid'};

for i = 1:length(fractions)
    t = fractions(i);
    base_pt = start_pt + t*(goal_pt-start_pt);

    % lateral direction perpendicular to path (in XY plane)
    perp = [-dir_vec(2), dir_vec(1), 0];
    obs_center = base_pt + lateral_offsets(i)*perp;

    switch types{i}
        case 'cuboid'
            lx = 90 + 20*rand();
            ly = 90 + 20*rand();
            h  = 150 + 150*rand();
            env = env.add_cuboid(obs_center(1), obs_center(2), 0, lx, ly, h, 15);
        case 'cylinder'
            r = 25 + 15*rand();
            h = 200 + 200*rand();
            env = env.add_cylinder(obs_center(1), obs_center(2), 0, r, h, 15);
    end
end

% Add a few extra scattered obstacles for visual density
extra_positions = [
    600,  1400, 0;
    1400, 400,  0;
    300,  1000, 0;
    1700, 1200, 0
];
extra_types = {'cylinder','cuboid','cylinder','cuboid'};
for i = 1:size(extra_positions,1)
    p = extra_positions(i,:);
    if strcmp(extra_types{i}, 'cylinder')
        env = env.add_cylinder(p(1), p(2), 0, 30, 250, 15);
    else
        env = env.add_cuboid(p(1), p(2), 0, 100, 100, 200, 15);
    end
end

fprintf('Environment: %d obstacles placed along and around direct corridor\n\n', ...
    numel(env.obstacles));

%% Step 2 — Run A* (finer resolution since obstacles are denser now)
[waypoints, raw_path, n_exp] = astar_3d(env, 40);

min_waypoints = 8;
if size(waypoints,1) < min_waypoints
    fprintf('Pruned waypoints (%d) too few — resampling at %d points.\n', ...
        size(waypoints,1), min_waypoints);
    waypoints = resample_path_arclength(raw_path, min_waypoints);
end
fprintf('Using %d waypoints for B-Spline fitting.\n', size(waypoints,1));

%% Step 3 — Smooth with B-Spline
fprintf('\nBuilding adaptive B-Spline over A*-derived waypoints...\n');
ga_knots = knot_ga(waypoints, 0.5, 30, 20);
smooth_path = bspline_engine(waypoints, 'chord', ga_knots);
fprintf('Smoothed path: %.1f m, max kappa=%.4f\n', ...
    smooth_path.length(), smooth_path.max_curvature());

raw_len = sum(sqrt(sum(diff(raw_path).^2,2)));
fprintf('Baseline (raw A*) length: %.1f m\n', raw_len);
fprintf('Length reduction: %.1f%%\n\n', ...
    100*(raw_len-smooth_path.length())/raw_len);

%% Step 4 — Verify no collisions after smoothing (critical safety check)
smooth_pts_check = smooth_path.sample(1000);
collision_found = false;
for i = 1:size(smooth_pts_check,1)
    if env.is_collision(smooth_pts_check(i,:))
        collision_found = true;
        fprintf('WARNING: smoothed path collides near point %d: [%.0f %.0f %.0f]\n', ...
            i, smooth_pts_check(i,:));
    end
end
if ~collision_found
    fprintf('Safety check PASSED: smoothed path clears all obstacles+buffers.\n\n');
else
    fprintf('Safety check FAILED — consider tightening kappa_max in knot_ga.\n\n');
end

%% Step 5 — Static comparison plot
results_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');

fig1 = figure('Position',[50 50 1400 950]);
ax = axes(fig1);
env.plot_obstacles(ax);
hold(ax,'on');

plot3(ax, raw_path(:,1), raw_path(:,2), raw_path(:,3), ...
    'b--', 'LineWidth',1.8, 'DisplayName','Baseline A* Path');
smooth_pts = smooth_path.sample(500);
plot3(ax, smooth_pts(:,1), smooth_pts(:,2), smooth_pts(:,3), ...
    'r-', 'LineWidth',3, 'DisplayName','Adaptive B-Spline (GA)');
plot3(ax, waypoints(:,1), waypoints(:,2), waypoints(:,3), ...
    'ko', 'MarkerSize',7, 'MarkerFaceColor','y', 'DisplayName','Waypoints');
plot3(ax, start_pt(1),start_pt(2),start_pt(3), ...
    'g^','MarkerSize',14,'MarkerFaceColor','g','DisplayName','Start');
plot3(ax, goal_pt(1),goal_pt(2),goal_pt(3), ...
    'rv','MarkerSize',14,'MarkerFaceColor','r','DisplayName','Goal');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Part 4 (Dense) — UAV Navigation Through %d Obstacles', ...
    numel(env.obstacles)));
legend('Location','best'); grid on; view(50,28); axis equal;
xlim([0 2000]); ylim([0 2000]); zlim([0 500]);

saveas(fig1, fullfile(results_dir, 'matlab_part4_dense_static.png'));
fprintf('Saved: matlab_part4_dense_static.png\n');

%% Step 6 — Fly the path
fprintf('\nRunning attitude controller along dense obstacle field...\n');

Ixx=6.5e-4; Iyy=6.5e-4; Izz=1.2e-3;
I = diag([Ixx,Iyy,Izz]);
Pq=20; Pw=4; tau_max=4.0; noise_amp=0.1;
dt = 1e-4;

u_lo = smooth_path.knot_vec(smooth_path.degree+1);
u_hi = smooth_path.knot_vec(end-smooth_path.degree) - 1e-10;

q = tangent_to_quat_p4(smooth_path.tangent_at(u_lo+0.001), [0;0;1]);
w = [0;0;0];
u_current = u_lo + 0.001;
u_step = 5e-6;
err_threshold = 0.1;

log_every = 500;
T_max = 60; steps = round(T_max/dt);
n_log = ceil(steps/log_every)+1;

pos_log = zeros(n_log,3); li=0;

for i = 1:steps
    tang  = smooth_path.tangent_at(min(u_current, u_hi-1e-6));
    q_ref = tangent_to_quat_p4(tang, [0;0;1]);
    noise = noise_amp*(rand(3,1)-0.5)*2;
    [tau, ~, err_mag] = quaternion_utils.p2_control(q_ref, q, w+noise, Pq, Pw, tau_max);
    [q, w] = quaternion_utils.rk4_step(q, w, tau, I, dt);

    if err_mag < err_threshold && u_current < u_hi
        u_current = min(u_current + u_step, u_hi);
    end

    if mod(i, log_every)==0 || i==steps
        li = li+1;
        pos_log(li,:) = smooth_path.evaluate(u_current);
    end
    if u_current >= u_hi - 1e-6
        fprintf('Flight complete at t=%.2fs\n', i*dt);
        break;
    end
end
pos_log = pos_log(1:li,:);

%% Step 7 — Animation
fprintf('\nBuilding animation...\n');
fig2 = figure('Position',[50 50 1300 900]);
ax2 = axes(fig2);
env.plot_obstacles(ax2);
hold(ax2,'on');
plot3(ax2, smooth_pts(:,1), smooth_pts(:,2), smooth_pts(:,3), ...
    'r-', 'LineWidth',1.5);
plot3(ax2, start_pt(1),start_pt(2),start_pt(3),'g^','MarkerSize',10,'MarkerFaceColor','g');
plot3(ax2, goal_pt(1),goal_pt(2),goal_pt(3),'rv','MarkerSize',10,'MarkerFaceColor','r');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('UAV Flight Through %d Obstacles', numel(env.obstacles)));
grid on; view(50,28); axis equal;
xlim([0 2000]); ylim([0 2000]); zlim([0 500]);

drone_marker = plot3(ax2, pos_log(1,1), pos_log(1,2), pos_log(1,3), ...
    'bo', 'MarkerSize',12, 'MarkerFaceColor','b');
trail = animatedline(ax2, 'Color','c', 'LineWidth',2);

vid_path = fullfile(results_dir, 'matlab_part4_dense_animation.mp4');
v = VideoWriter(vid_path, 'MPEG-4');
v.FrameRate = 20;
open(v);

frame_skip = max(1, floor(size(pos_log,1)/300));
for i = 1:frame_skip:size(pos_log,1)
    set(drone_marker, 'XData',pos_log(i,1), 'YData',pos_log(i,2), 'ZData',pos_log(i,3));
    addpoints(trail, pos_log(i,1), pos_log(i,2), pos_log(i,3));
    drawnow;
    writeVideo(v, getframe(fig2));
end
close(v);
fprintf('Saved animation: matlab_part4_dense_animation.mp4\n');

fprintf('\n=== Part 4 (Dense) Summary ===\n');
fprintf('Obstacles: %d\n', numel(env.obstacles));
fprintf('A* nodes expanded: %d\n', n_exp);
fprintf('Baseline path length: %.1f m\n', raw_len);
fprintf('Smoothed path length: %.1f m\n', smooth_path.length());
fprintf('Length reduction: %.1f%%\n', 100*(raw_len-smooth_path.length())/raw_len);
fprintf('Max curvature: %.4f\n', smooth_path.max_curvature());
fprintf('Obstacles cleared: %s\n', string(~collision_found));

%% ── Local helpers ──────────────────────────────────────

function q = tangent_to_quat_p4(tangent, up)
    tn = tangent(:) / max(norm(tangent), 1e-8);
    if abs(dot(tn, up)) > 0.95, up = [0;1;0]; end
    y = cross(up, tn); y = y/norm(y);
    z = cross(tn, y);
    R = [tn, y, z];
    tr = trace(R);
    if tr > 0
        s = 0.5/sqrt(tr+1);
        q = [0.25/s; (R(3,2)-R(2,3))*s; (R(1,3)-R(3,1))*s; (R(2,1)-R(1,2))*s];
    else
        q = [1;0;0;0];
    end
    q = q/norm(q);
end

function pts = resample_path_arclength(raw_path, n_points)
    d = diff(raw_path);
    seg_len = sqrt(sum(d.^2,2));
    cum_len = [0; cumsum(seg_len)];
    total = cum_len(end);
    target = linspace(0, total, n_points);
    pts = zeros(n_points, 3);
    for i = 1:n_points
        idx = find(cum_len >= target(i), 1);
        if idx == 1
            pts(i,:) = raw_path(1,:);
        else
            t0 = cum_len(idx-1); t1 = cum_len(idx);
            frac = (target(i)-t0) / max(t1-t0, 1e-10);
            pts(i,:) = raw_path(idx-1,:) + frac*(raw_path(idx,:) - raw_path(idx-1,:));
        end
    end
    pts(1,:)   = raw_path(1,:);
    pts(end,:) = raw_path(end,:);
end