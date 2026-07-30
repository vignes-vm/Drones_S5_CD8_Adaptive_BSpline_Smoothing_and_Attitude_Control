%% TEST_STEP_RESPONSE  Paper 2 Figure 3 Recreation
% FIX: control law recomputed at the SAME rate as physics integration.
% This system's fast pole has a 160-microsecond time constant
% (verified analytically: Pq=20, Pw=4, I=6.5e-4 gives poles at
% -5.4 rad/s and -6150 rad/s). Holding torque constant for longer
% than that lets omega build up unbounded before the controller
% reacts -- that was the actual bug, not the RK4 integrator itself.

clear; clc; close all;
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));

Ixx = 6.5e-4; Iyy = 6.5e-4; Izz = 1.2e-3;
I   = diag([Ixx, Iyy, Izz]);
Pq  = 20; Pw = 4;
tau_max   = 4.0;
noise_amp = 0.1;

dt   = 1e-4;         % control AND physics run at this rate now
T    = 15;
steps = round(T/dt);

log_every = 100;      % downsample for plotting/printing only
n_log = ceil(steps/log_every);

q = [1;0;0;0]; w = [0;0;0];

t_h   = zeros(1,n_log);
e_h   = zeros(3,n_log);
r_h   = zeros(3,n_log);
tau_h = zeros(3,n_log);
li    = 0;

fprintf('Running Paper 2 step response (control @ %.0fHz)...\n', 1/dt);

for i = 1:steps
    t = (i-1)*dt;

    if     t >= 11.0, q_ref = quaternion_utils.euler2q(0,0,1.0);
    elseif t >=  6.0, q_ref = quaternion_utils.euler2q(0,1.0,0);
    elseif t >=  2.0, q_ref = quaternion_utils.euler2q(1.0,0,0);
    else,             q_ref = [1;0;0;0];
    end

    noise = noise_amp*(rand(3,1)-0.5)*2;
    [tau, axis_err] = quaternion_utils.p2_control(...
        q_ref, q, w+noise, Pq, Pw, tau_max);

    % Control recomputed EVERY step now -- no frozen-torque window
    [q, w] = quaternion_utils.rk4_step(q, w, tau, I, dt);

    if mod(i, log_every) == 0 || i == steps
        li = li + 1;
        t_h(li)     = t;
        e_h(:,li)   = quaternion_utils.q2euler(q);
        r_h(:,li)   = quaternion_utils.q2euler(q_ref);
        tau_h(:,li) = tau;
    end

    if mod(i, round(steps/10))==0
        fprintf('  %3.0f%% | t=%.1fs | err=%.4f\n', 100*i/steps, t, norm(axis_err));
    end
end

t_h = t_h(1:li); e_h = e_h(:,1:li); r_h = r_h(:,1:li); tau_h = tau_h(:,1:li);

figure('Position',[100 100 900 700]);
labels = {'\phi (rad)','\theta (rad)','\psi (rad)'};
cols = {'b','g','r'};
for ax = 1:3
    subplot(3,1,ax);
    plot(t_h, r_h(ax,:), 'k--', 'LineWidth',1.5, 'DisplayName','Reference'); hold on;
    plot(t_h, e_h(ax,:), cols{ax}, 'LineWidth',1.5, 'DisplayName','Actual');
    ylabel(labels{ax},'FontSize',11); ylim([-0.5 1.5]); grid on;
    legend('Location','northeast','FontSize',9);
end
xlabel('Time (s)','FontSize',11);
subplot(3,1,1); title('Step Response — Paper 2 Figure 3','FontSize',12);

results_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results');
saveas(gcf, fullfile(results_dir, 'matlab_paper2_step.png'));
fprintf('Saved: matlab_paper2_step.png\n');

fprintf('\nMetrics:\n');
for ax = 1:3
    fprintf('  Axis %d final error: %.4f rad\n', ax, abs(e_h(ax,end)-r_h(ax,end)));
end
fprintf('  Max torque: %.3f Nm\n', max(abs(tau_h(:))));