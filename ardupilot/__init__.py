"""
ArduPilot SITL Integration Package.
Connects the Adaptive B-Spline Path Smoother and Quaternion Attitude Guidance to ArduPilot SITL.
"""

from ardupilot.sitl_bridge import ArduPilotBridge, main

__all__ = ["ArduPilotBridge", "main"]
