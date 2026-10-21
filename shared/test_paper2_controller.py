"""
Standalone tests for the Paper 2 quaternion P^2 attitude controller —
no path planning involved. Reproduces Paper 2's own simulation
scenarios (step response, sinusoidal tracking, flip maneuver) to
validate the controller in isolation.
"""
import matplotlib
matplotlib.use('Agg')  # Fix macOS crash — MUST be first
import matplotlib.pyplot as plt
import numpy as np

from attitude_control.quadrotor_model import QuadrotorModel
from attitude_control.p2_controller import P2Controller
from attitude_control.quaternion import Quaternion

# Paper 2 exact parameters
Pq        = 20.0
Pw        = 4.0
NOISE_STD = 0.1
DT        = 0.01
TAU_LIMIT = 4.0


def compute_settling_time(t, err_mag, threshold=0.05):
    """Time after which err_mag stays below threshold for the rest of the run."""
    above = np.where(err_mag >= threshold)[0]
    if len(above) == 0:
        return 0.0
    last_bad = above[-1]
    if last_bad + 1 >= len(t):
        return float('nan')
    return t[last_bad + 1]


def run_sim(q_ref_func, total_time, dt=DT, noise_std=NOISE_STD):
    drone = QuadrotorModel()
    ctrl  = P2Controller(Pq=Pq, Pw=Pw)
    n_steps = int(total_time / dt)

    t_arr        = np.zeros(n_steps)
    euler_actual = np.zeros((n_steps, 3))
    euler_ref    = np.zeros((n_steps, 3))
    torque_arr   = np.zeros((n_steps, 3))
    err_mag_arr  = np.zeros(n_steps)
    nan_detected = False

    for i in range(n_steps):
        t = i * dt
        q_ref = q_ref_func(t)

        omega_noisy = drone.omega + np.random.normal(0, noise_std, 3)

        tau, axis_err = ctrl.compute_torque(q_ref, drone.q, omega_noisy)
        drone.step_attitude_only(tau, dt)

        t_arr[i]        = t
        euler_actual[i] = drone.q.to_euler()
        euler_ref[i]    = q_ref.to_euler()
        torque_arr[i]   = tau
        err_mag_arr[i]  = np.linalg.norm(axis_err)

        if np.any(np.isnan(drone.q.q)) or np.any(np.isnan(tau)):
            nan_detected = True

    return {
        't':            t_arr,
        'euler':        euler_actual,
        'euler_ref':    euler_ref,
        'torque':       torque_arr,
        'err_mag':      err_mag_arr,
        'nan_detected': nan_detected,
    }


def report(name, result, unwrap_axes=None):
    t       = result['t']
    err_mag = result['err_mag']
    torque  = result['torque']

    max_err     = np.max(err_mag)
    max_torque  = np.max(np.abs(torque))
    settling    = compute_settling_time(t, err_mag)
    no_nan      = not result['nan_detected']

    settling_str = f"{settling:.2f}s" if not np.isnan(settling) else "never (did not settle)"

    print(f"\n--- {name} ---")
    print(f"  Max tracking error: {max_err:.4f}")
    print(f"  Settling time:      {settling_str}")
    print(f"  Max torque used:    {max_torque:.3f} Nm")
    print(f"  Flip / NaN check:   {'OK (no NaN)' if no_nan else 'FAILED (NaN detected)'}")


def plot_result(result, title, filename, unwrap_axes=None):
    t     = result['t']
    eu    = result['euler'].copy()
    er    = result['euler_ref'].copy()

    if unwrap_axes:
        for ax_idx in unwrap_axes:
            eu[:, ax_idx] = np.unwrap(eu[:, ax_idx])
            er[:, ax_idx] = np.unwrap(er[:, ax_idx])

    labels = ['Roll φ (rad)', 'Pitch θ (rad)', 'Yaw ψ (rad)']
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    for i in range(3):
        axes[i].plot(t, er[:, i], 'k--', lw=1.5, alpha=0.8, label='Reference')
        axes[i].plot(t, eu[:, i], lw=1.5, label='Actual')
        axes[i].set_ylabel(labels[i], fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=9, loc='upper right')

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ════════════════════════════════════════
# TEST 1 — Step Response (Paper 2 Figure 3)
# ════════════════════════════════════════
def step_q_ref(t):
    if t < 1.0:
        return Quaternion()
    elif t < 5.0:
        return Quaternion.from_euler(1.0, 0.0, 0.0)
    elif t < 10.0:
        return Quaternion.from_euler(0.0, 1.0, 0.0)
    else:
        return Quaternion.from_euler(0.0, 0.0, 1.0)


# ════════════════════════════════════════
# TEST 2 — Sinusoidal Tracking (Paper 2 Figure 5)
# ════════════════════════════════════════
def sine_q_ref(t):
    phi   = 0.5 * np.sin(1.0 * t)
    theta = 0.5 * np.sin(1.0 * t + 2 * np.pi / 3)
    psi   = 0.5 * np.sin(1.0 * t + 4 * np.pi / 3)
    return Quaternion.from_euler(phi, theta, psi)


# ════════════════════════════════════════
# TEST 3 — Flip Maneuver (Paper 2 Figure 7)
# ════════════════════════════════════════
FLIP_TOTAL_TIME = 5.0

def flip_q_ref(t):
    phi = 2 * np.pi * (t / FLIP_TOTAL_TIME)
    return Quaternion.from_euler(phi, 0.0, 0.0)


if __name__ == '__main__':
    print("=" * 55)
    print("PAPER 2 QUATERNION CONTROLLER — STANDALONE TESTS")
    print(f"Pq={Pq}, Pw={Pw}, noise_std={NOISE_STD}, "
          f"torque bounds=±{TAU_LIMIT}Nm")
    print("=" * 55)

    result1 = run_sim(step_q_ref, total_time=15.0)
    report("TEST 1: Step Response", result1)
    plot_result(
        result1,
        'Step Response — Quaternion P² Controller (Paper 2 Fig. 3)',
        'results/test_step_response.png'
    )

    result2 = run_sim(sine_q_ref, total_time=15.0)
    report("TEST 2: Sinusoidal Tracking", result2)
    plot_result(
        result2,
        'Sinusoidal Tracking — Quaternion P² Controller (Paper 2 Fig. 5)',
        'results/test_sine_tracking.png'
    )

    result3 = run_sim(flip_q_ref, total_time=FLIP_TOTAL_TIME)
    report("TEST 3: Flip Maneuver", result3)
    plot_result(
        result3,
        '360° Flip Maneuver — Quaternion P² Controller (Paper 2 Fig. 7)',
        'results/test_flip.png',
        unwrap_axes=[0]
    )

    print("\n" + "=" * 55)
    print("ALL TESTS COMPLETE")
    print("=" * 55)
