%% RUN_MERGED_SIMULATION  Full pipeline: B-Spline path -> Quaternion control
% Builds on validated Part 1 (bspline_engine) and Part 2 (quaternion_utils)

clear; clc; close all;
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

fprintf('=== Merged Simulation: Paper 1 Path + Paper 2 Controller ===\n\n');

%% Step 1 — Build the adaptive B-Spline path (reuse Part 1)
waypoints = [
    0,    0,    20;
    250,  150,  60;
    500,  80,   110;
    750,  300,  90;
    1000, 250,  130;
    1250, 500,  155;
    1500, 800,  170;
    1750, 1300, 185;
    2000, 1800, 200
];

fprintf('Building adaptive B-Spline path...\n');
ga_knots = knot_ga(waypoints, 0.5, 30, 20);
path = bspline_engine(waypoints, 'chord', ga_knots);
fprintf('Path ready: %.1f m, max kappa=%.4f\n\n', ...
    path.length(), path.max_curvature());

%% Step 2 — Controller and dynamics setup (validated Part 2 parameters)
Ixx = 6.5e-4; Iyy = 6.5e-4; Izz = 1.2e-3;
I   = diag([Ixx, Iyy, Izz]);
Pq  = 20; Pw = 4;
tau_max   = 4.0;
noise_amp = 0.1;

dt = 1e-4;            % validated rate from Part 2 — do not increase
log_every = 200;       % downsample logging only, not control

%% Step 3 — Reference generation from path tangent
u_lo = path.knot_vec(path.degree+1);
u_hi = path.knot_vec(end-path.degree) - 1e-10;

function q = tangent_to_quat(tangent, up)
    tn = tangent / max(norm(tangent), 1e-8);
    if abs(dot(tn, up)) > 0.95
        up = [0;1;0];
    end
    y = cross(up, tn); y = y/norm(y);
    z = cross(tn, y);
    R = [tn(:), y(:), z(:)];
    tr = trace(R);
    if tr > 0
        s = 0.5/sqrt(tr+1);
        q = [0.25/s; (R(3,2)-R(2,3))*s; (R(1,3)-R(3,1))*s; (R(2,1)-R(1,2))*s];
    else
        q = [1;0;0;0];
    end
    q = q/norm(q);
end

%% Step 4 — Main simulation loop
q = [1;0;0;0];
w = [0;0;0];

% Start aligned with initial path tangent
tang0 = path.tangent_at(u_lo+0.001);
q = tangent_to_quat(tang0, [0;0;1]);

u_current = u_lo + 0.001;
u_step    = 5e-6;        % path parameter advance per control step
                          % (tuned small since dt is small)
err_threshold = 0.1;

T_max = 60;              % safety cap
steps = round(T_max/dt);
n_log = ceil(steps/log_every)+1;

t_h = zeros(1,n_log); e_h = zeros(3,n_log); r_h = zeros(3,n_log);
tau_h = zeros(3,n_log); u_h = zeros(1,n_log); err_h = zeros(1,n_log);
li = 0;

fprintf('Running merged simulation (control @ %.0fHz)...\n', 1/dt);

for i = 1:steps
    t = (i-1)*dt;

    tang  = path.tangent_at(min(u_current, u_hi-1e-6));
    q_ref = tangent_to_quat(tang, [0;0;1]);

    noise = noise_amp*(rand(3,1)-0.5)*2;
    [tau, axis_err, err_mag] = quaternion_utils.p2_control(...
        q_ref, q, w+noise, Pq, Pw, tau_max);

    [q, w] = quaternion_utils.rk4_step(q, w, tau, I, dt);

    % Advance along path only when tracking is good (mirrors real
    % guidance logic: don't move forward until pointed correctly)
    if err_mag < err_threshold && u_current < u_hi
        u_current = min(u_current + u_step, u_hi);
    end

    if mod(i, log_every) == 0 || i == steps
        li = li+1;
        t_h(li) = t; e_h(:,li) = quaternion_utils.q2euler(q);
        r_h(:,li) = quaternion_utils.q2euler(q_ref);
        tau_h(:,li) = tau; u_h(li) = u_current; err_h(li) = err_mag;
    end

    if mod(i, round(steps/20))==0
        fprintf('  t=%5.1fs | u=%.4f/%.4f | err=%.4f\n', ...
            t, u_current, u_hi, err_mag);
    end

    if u_current >= u_hi - 1e-6
        fprintf('\nPath complete at t=%.2fs\n', t);
        break;
    end
end

n = li;
t_h=t_h(1:n); e_h=e_h(:,1:n); r_h=r_h(:,1:n);
tau_h=tau_h(:,1:n); u_h=u_h(1:n); err_h=err_h(1:n);

%% Step 5 — Plots
results_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');

figure('Position',[100 100 1000 800]);
labels = {'\phi','\theta','\psi'}; cols = {'b','g','r'};
for ax=1:3
    subplot(3,1,ax);
    plot(t_h, r_h(ax,:), 'k--', 'DisplayName','Reference'); hold on;
    plot(t_h, e_h(ax,:), cols{ax}, 'DisplayName','Actual');
    ylabel(labels{ax}); grid on; legend;
end
xlabel('Time (s)');
subplot(3,1,1); title('Merged Simulation — Attitude Tracking Along B-Spline Path');
saveas(gcf, fullfile(results_dir,'matlab_merged_attitude.png'));

figure('Position',[100 100 900 400]);
plot(t_h, err_h, 'Color', [0.5 0 0.5], 'LineWidth',1.2); grid on;
yline(err_threshold,'r--','advance threshold');
xlabel('Time (s)'); ylabel('||q_{err}||'); title('Tracking Error');
saveas(gcf, fullfile(results_dir,'matlab_merged_error.png'));

figure('Position',[100 100 900 400]);
plot(t_h, u_h, 'Color', [0 0 0.5], 'LineWidth',1.5); grid on;
xlabel('Time (s)'); ylabel('Path parameter u'); title('Path Progress');
saveas(gcf, fullfile(results_dir,'matlab_merged_progress.png'));

fprintf('\nSaved 3 plots to results/\n');
fprintf('Final path progress: %.1f%%\n', 100*(u_h(end)-u_lo)/(u_hi-u_lo));
fprintf('Mean tracking error: %.4f\n', mean(err_h));
fprintf('Max torque used: %.3f Nm\n', max(abs(tau_h(:))));