"""
Master Simulation Entry Point:
Replicates and Simulates Both Papers in MuJoCo Physics Engine:

Paper 1: "Full Quaternion Based Attitude Control for a Quadrotor"
          (Emil Fresk and George Nikolakopoulos, ECC 2013)
Paper 2: "An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning"
          (Mykola Nikolaiev and Mykhailo Novotarskyi, IJISA 2025)

Supports live 3D MuJoCo GUI window rendering with --gui.
"""

import os
import sys
import time
import argparse

from simulate_paper1 import run_paper1_simulations
from simulate_paper2 import run_paper2_simulations

def main():
    parser = argparse.ArgumentParser(description='Master Simulation Suite')
    parser.add_argument('--gui', action='store_true', help='Launch live 3D interactive MuJoCo GUI window')
    args = parser.parse_args()

    t_start = time.time()
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    print("========================================================================")
    print("      MUJOCO SIMULATION SUITE FOR QUADROTOR CONTROL & PATH PLANNING     ")
    print("========================================================================")
    print(" Paper 1: Full Quaternion Based Attitude Control for a Quadrotor")
    print(" Paper 2: An Enhanced Adaptive B-spline Smoothing Approach for UAV Path Planning")
    print(f" 3D GUI Window: {'ENABLED' if args.gui else 'DISABLED (Use --gui to open)'}")
    print("========================================================================\n")

    # Step 1: Run Paper 1 Simulation
    run_paper1_simulations(gui=args.gui, output_dir=output_dir)

    # Step 2: Run Paper 2 Simulation
    run_paper2_simulations(gui=args.gui, output_dir=output_dir)

    t_total = time.time() - t_start
    print(f"\n[SUCCESS] All Paper 1 & Paper 2 simulations completed in {t_total:.2f} seconds!")
    print(f"Generated Plot Artifacts in directory: '{os.path.abspath(output_dir)}':")

    plots = [
        "paper1_fig3_step_response.png",
        "paper1_fig4_step_torques.png",
        "paper1_fig5_sine_tracking.png",
        "paper1_fig6_sine_torques.png",
        "paper1_fig7_flip_maneuver.png",
        "paper1_fig8_flip_quaternions.png",
        "paper2_3d_trajectories.png",
        "paper2_curvature_comparison.png",
        "paper2_mujoco_flight_tracking.png",
        "paper2_metrics_summary.txt"
    ]

    for p in plots:
        full_p = os.path.join(output_dir, p)
        exists = os.path.exists(full_p)
        status = "EXISTS" if exists else "MISSING"
        print(f" - [{status}] {p}")

if __name__ == '__main__':
    main()
