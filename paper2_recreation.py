"""
Recreation of Fresk & Nikolakopoulos, "Full Quaternion Based Attitude
Control for a Quadrotor" (ECC 2013) — Figures 3, 5, 7.

Uses the paper's exact parameters (Ixx=Iyy=6.5e-4, Izz=1.2e-3 kg.m^2,
Pq=20, Pw=4, noise amplitude 0.1, torque bounds +/-4 Nm). Results are
logged/plotted at the paper's dt=0.01s resolution, but the control law
and physics are integrated internally at a finer dt=0.0005s (2kHz).

Why the finer internal rate: at dt=0.01 (100 Hz), Pq=20/Pw=4 combined
with this project's very low inertia (Ixx=6.5e-4) makes the discrete
control loop unstable — 4Nm of torque on that inertia is ~6154 rad/s^2
of angular acceleration, and a 100 Hz update overshoots into permanent
torque-saturation chatter (verified separately: err stays pinned near
1.0 and never converges at dt=0.01). Running the same control law and
physics at 2kHz internally removes that discretization artifact while
keeping every physical parameter identical to the paper's, and gives
converging step/sine/flip responses that can be logged at the paper's
own 0.01s resolution for the plots below.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from attitude_control.quaternion import Quaternion

# Paper 2 exact parameters
Pq         = 20.0
Pw         = 4.0
NOISE_STD  = 0.1
TAU_LIMIT  = 4.0
DT_LOG     = 0.01     # paper's logging / control-command resolution
DT_PHYSICS = 0.0005   # internal control+physics update rate (2kHz)


def run_sim(q_ref_func, total_time, dt_log=DT_LOG,
            dt_physics=DT_PHYSICS, noise_std=NOISE_STD):
    drone = QuadrotorModel()
    ctrl  = P2Controller(Pq=Pq, Pw=Pw)

    substeps = int(round(dt_log / dt_physics))
    n_log    = int(total_time / dt_log)

    t_arr        = np.zeros(n_log)
    euler_actual = np.zeros((n_log, 3))
    euler_ref    = np.zeros((n_log, 3))
    torque_arr   = np.zeros((n_log, 3))
    err_mag_arr  = np.zeros(n_log)
    q_arr        = np.zeros((n_log, 4))
    nan_detected = False

    for i in range(n_log):
        t     = i * dt_log
        q_ref = q_ref_func(t)

        for _ in range(substeps):
            omega_noisy = drone.omega + np.random.normal(0, noise_std, 3)
            tau, axis_err = ctrl.compute_torque(q_ref, drone.q, omega_noisy)
            drone.step_attitude_only(tau, dt_physics)

        t_arr[i]        = t
        euler_actual[i] = drone.q.to_euler()
        euler_ref[i]    = q_ref.to_euler()
        torque_arr[i]   = tau
        err_mag_arr[i]  = np.linalg.norm(axis_err)
        q_arr[i]        = drone.q.q.copy()

        if np.any(np.isnan(drone.q.q)) or np.any(np.isnan(tau)):
            nan_detected = True

    return {
        't':            t_arr,
        'euler':        euler_actual,
        'euler_ref':    euler_ref,
        'torque':       torque_arr,
        'err_mag':      err_mag_arr,
        'q':            q_arr,
        'nan_detected': nan_detected,
    }


def compute_metrics(result, threshold=0.05):
    t   = result['t']
    err = result['err_mag']

    max_err  = np.max(err)
    rms_err  = np.sqrt(np.mean(err ** 2))
    max_tau  = np.max(np.abs(result['torque']))
    nan_flag = result['nan_detected']

    above = np.where(err >= threshold)[0]
    if len(above) == 0:
        settling = 0.0
    elif above[-1] + 1 >= len(t):
        settling = float('nan')
    else:
        settling = t[above[-1] + 1]

    return {
        'max_err': max_err,
        'rms_err': rms_err,
        'settling': settling,
        'max_tau': max_tau,
        'nan_flag': nan_flag,
    }


def report(name, metrics):
    settling_str = (f"{metrics['settling']:.2f}s"
                     if not np.isnan(metrics['settling'])
                     else "never (did not settle)")
    print(f"\n--- {name} ---")
    print(f"  Max tracking error:  {metrics['max_err']:.4f} rad")
    print(f"  RMS tracking error:  {metrics['rms_err']:.4f} rad")
    print(f"  Settling time:       {settling_str}")
    print(f"  Max torque used:     {metrics['max_tau']:.3f} Nm")
    print(f"  NaN detected:        {'YES' if metrics['nan_flag'] else 'NO'}")


# ════════════════════════════════════════
# TEST CASE 1 — Step Response (Figure 3)
# ════════════════════════════════════════
def step_q_ref(t):
    if t < 2.0:
        return Quaternion()
    elif t < 6.0:
        return Quaternion.from_euler(1.0, 0.0, 0.0)
    elif t < 11.0:
        return Quaternion.from_euler(0.0, 1.0, 0.0)
    else:
        return Quaternion.from_euler(0.0, 0.0, 1.0)


def plot_step_response(result, filename):
    t  = result['t']
    eu = np.unwrap(result['euler'],     axis=0)
    er = np.unwrap(result['euler_ref'], axis=0)
    labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for i in range(3):
        axes[i].plot(t, er[:, i], 'k--', lw=1.5, label='Reference')
        axes[i].plot(t, eu[:, i], lw=1.5, label='Actual')
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Paper 2 Recreation — Step Response (Figure 3)', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ════════════════════════════════════════
# TEST CASE 2 — Sinusoidal Tracking (Figure 5)
# ════════════════════════════════════════
def sine_q_ref(t):
    phi   = 0.5 * np.sin(1.0 * t)
    theta = 0.5 * np.sin(1.0 * t + np.pi / 4)
    psi   = 0.5 * np.sin(1.0 * t + np.pi / 2)
    return Quaternion.from_euler(phi, theta, psi)


def plot_sine_tracking(result, filename):
    t  = result['t']
    eu = result['euler']
    er = result['euler_ref']
    labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for i in range(3):
        axes[i].plot(t, er[:, i], 'k--', lw=1.5, label='Reference')
        axes[i].plot(t, eu[:, i], lw=1.5, label='Actual')
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Paper 2 Recreation — Sinusoidal Tracking (Figure 5)', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ════════════════════════════════════════
# TEST CASE 3 — 360 Degree Flip (Figure 7)
# ════════════════════════════════════════
FLIP_TOTAL_TIME = 5.0

def flip_q_ref(t):
    phi = 2 * np.pi * (t / FLIP_TOTAL_TIME)
    return Quaternion.from_euler(phi, 0.0, 0.0)


def plot_flip(result, filename):
    t   = result['t']
    roll_actual = np.unwrap(result['euler'][:, 0])
    roll_ref    = np.unwrap(result['euler_ref'][:, 0])
    q0 = result['q'][:, 0]
    q1 = result['q'][:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, roll_ref, 'k--', lw=1.5, label='Reference (ramp)')
    axes[0].plot(t, roll_actual, 'tomato', lw=1.5, label='Actual roll')
    axes[0].set_ylabel('Roll φ (rad)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].set_title('360° Flip — Roll Angle')

    axes[1].plot(t, q0, color='steelblue', lw=1.5, label='q0 (scalar)')
    axes[1].plot(t, q1, color='seagreen',  lw=1.5, label='q1 (x-axis)')
    axes[1].axhline(y=0, color='k', linestyle='--', lw=0.8, alpha=0.5)
    axes[1].set_ylabel('Quaternion component')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_title('360° Flip — Quaternion Components (no singularity)')

    fig.suptitle('Paper 2 Recreation — 360° Flip Maneuver (Figure 7)', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


if __name__ == '__main__':
    print("=" * 60)
    print("PAPER 2 RECREATION — Fresk & Nikolakopoulos, ECC 2013")
    print(f"Pq={Pq}, Pw={Pw}, noise_std={NOISE_STD}, "
          f"torque bounds=±{TAU_LIMIT}Nm")
    print(f"Logged at dt={DT_LOG}s (paper resolution), "
          f"integrated internally at dt={DT_PHYSICS}s")
    print("=" * 60)

    result1 = run_sim(step_q_ref, total_time=15.0)
    m1 = compute_metrics(result1)
    report("TEST CASE 1: Step Response (Fig. 3)", m1)
    plot_step_response(result1, 'results/paper2_step_response.png')

    result2 = run_sim(sine_q_ref, total_time=15.0)
    m2 = compute_metrics(result2)
    report("TEST CASE 2: Sinusoidal Tracking (Fig. 5)", m2)
    plot_sine_tracking(result2, 'results/paper2_sine_tracking.png')

    result3 = run_sim(flip_q_ref, total_time=FLIP_TOTAL_TIME)
    m3 = compute_metrics(result3)
    report("TEST CASE 3: 360° Flip Maneuver (Fig. 7)", m3)
    plot_flip(result3, 'results/paper2_flip.png')

    print("\n" + "=" * 60)
    print("ALL TEST CASES COMPLETE")
    print("=" * 60)
