%% RUN_PAPER1  Recreate Table 2 from Paper 1
clear; clc; close all;
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

fprintf('=== Paper 1 Recreation (Fresh Build) ===\n\n');

start_pt = [0, 0, 20];
goal_pt  = [2000, 1800, 200];
straight_dist = norm(goal_pt - start_pt);

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

%% Method 1 — Baseline Polyline
fprintf('Method 1: Baseline Polyline\n');
diffs = diff(waypoints);
baseline_len = sum(sqrt(sum(diffs.^2, 2)));
baseline_SR  = baseline_len / straight_dist;
baseline_kap = polyline_curvature(waypoints);
fprintf('  Length=%.1f  kappa=%.4f  SR=%.3f\n\n', ...
    baseline_len, baseline_kap, baseline_SR);

%% Method 2 — Uniform B-Spline
fprintf('Method 2: Uniform B-Spline\n');
tic;
e_uniform = bspline_engine(waypoints, 'uniform');
t_u = toc;
len_u = e_uniform.length();
kap_u = e_uniform.max_curvature();
SR_u  = len_u / straight_dist;
fprintf('  Length=%.1f  kappa=%.4f  Time=%.3fs  SR=%.3f\n\n', ...
    len_u, kap_u, t_u, SR_u);

%% Method 3 — Classical B-Spline (Chord-Length)
fprintf('Method 3: Classical B-Spline (Chord-Length)\n');
tic;
e_chord = bspline_engine(waypoints, 'chord');
t_c = toc;
len_c = e_chord.length();
kap_c = e_chord.max_curvature();
SR_c  = len_c / straight_dist;
fprintf('  Length=%.1f  kappa=%.4f  Time=%.3fs  SR=%.3f\n\n', ...
    len_c, kap_c, t_c, SR_c);

%% Method 4 — Adaptive B-Spline (GA)
fprintf('Method 4: Adaptive B-Spline (GA)\n');
tic;
ga_knots = knot_ga(waypoints, 0.5, 30, 20);
e_adaptive = bspline_engine(waypoints, 'chord', ga_knots);
t_a = toc;
len_a = e_adaptive.length();
kap_a = e_adaptive.max_curvature();
SR_a  = len_a / straight_dist;
fprintf('  Length=%.1f  kappa=%.4f  Time=%.3fs  SR=%.3f\n\n', ...
    len_a, kap_a, t_a, SR_a);

%% Print Table
sep = repmat('=', 1, 68);
fprintf('%s\n', sep);
fprintf('Table 2: Performance of Four Path-Smoothing Methods\n');
fprintf('%s\n', repmat('-', 1, 68));
fprintf('%-25s %10s %12s %10s %8s\n', ...
    'Method','Length(m)','Max k(m-1)','Time(s)','SR');
fprintf('%s\n', repmat('-', 1, 68));
fprintf('%-25s %10.1f %12.4f %10.2f %8.3f\n', ...
    'Baseline Polyline',  baseline_len, baseline_kap, 0,   baseline_SR);
fprintf('%-25s %10.1f %12.4f %10.2f %8.3f\n', ...
    'Uniform B-Spline',   len_u,        kap_u,        t_u, SR_u);
fprintf('%-25s %10.1f %12.4f %10.2f %8.3f\n', ...
    'Classical B-Spline', len_c,        kap_c,        t_c, SR_c);
fprintf('%-25s %10.1f %12.4f %10.2f %8.3f\n', ...
    'Adaptive B-Spline',  len_a,        kap_a,        t_a, SR_a);
fprintf('%s\n', sep);
fprintf('Paper Reference Values:\n');
fprintf('%-25s %10.1f %12.2f %10.2f %8.3f\n','Baseline Polyline',2984.6,0.65,0.00,1.107);
fprintf('%-25s %10.1f %12.2f %10.2f %8.3f\n','Uniform B-Spline', 2840.8,0.56,0.25,1.053);
fprintf('%-25s %10.1f %12.2f %10.2f %8.3f\n','Classical B-Spline',2803.2,0.49,0.31,1.039);
fprintf('%-25s %10.1f %12.2f %10.2f %8.3f\n','Adaptive B-Spline',2731.4,0.38,0.42,1.013);
fprintf('%s\n\n', sep);

%% Plot
figure('Position', [100 100 1200 700]);
pts_u = e_uniform.sample(500);
pts_c = e_chord.sample(500);
pts_a = e_adaptive.sample(500);

plot3(waypoints(:,1),waypoints(:,2),waypoints(:,3),...
    'ko','MarkerSize',8,'MarkerFaceColor','k','DisplayName','Waypoints');
hold on;
plot3(pts_u(:,1),pts_u(:,2),pts_u(:,3),'b--','LineWidth',1.5,...
    'DisplayName',sprintf('Uniform L=%.0fm',len_u));
plot3(pts_c(:,1),pts_c(:,2),pts_c(:,3),'g-','LineWidth',1.5,...
    'DisplayName',sprintf('Classical L=%.0fm',len_c));
plot3(pts_a(:,1),pts_a(:,2),pts_a(:,3),'r-','LineWidth',2.5,...
    'DisplayName',sprintf('Adaptive L=%.0fm',len_a));
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Paper 1 — 3D Path Comparison (MATLAB)');
legend('Location','best'); grid on; view(45,30);

results_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');
saveas(gcf, fullfile(results_dir, 'matlab_paper1_paths.png'));
fprintf('Saved: matlab_paper1_paths.png\n');

%% Helper
function kap = polyline_curvature(pts)
    kap = 0;
    for i = 2:size(pts,1)-1
        v1 = pts(i,:)-pts(i-1,:);
        v2 = pts(i+1,:)-pts(i,:);
        n1 = norm(v1); n2 = norm(v2);
        if n1>1e-10 && n2>1e-10
            c = max(-1,min(1,dot(v1,v2)/(n1*n2)));
            kap = max(kap, acos(c)/(0.5*(n1+n2)));
        end
    end
end