# Project Proposal (FRP)

## Project Title

**Quaternion-Based Attitude and Trajectory Control of a UAV using P² Controller**

## Team Members

| Name | Roll Number | Email |
|---|---|---|
| Aparna A | CB.SC.U4AIE24206 | cb.sc.u4aie24206@cb.students.amrita.edu |
| G Prajwal Priyadarshan | CB.SC.U4AIE24214 | cb.sc.u4aie24214@cb.students.amrita.edu |
| Kabilan K | CB.SC.U4AIE24224 | cb.sc.u4aie24224@cb.students.amrita.edu |
| Kesav Satya Sai Nimmagadda | CB.SC.U4AIE24236 | cb.sc.u4aie24236@cb.students.amrita.edu |
| Vignes V M | CB.SC.U4AIE24359 | cb.sc.u4aie24359@cb.students.amrita.edu |

## Proposed Project Work

This project aims to develop a quaternion-based control framework for UAV attitude and trajectory control. The main focus is to design and simulate a quaternion-based P² attitude controller that can accurately control the rotational motion of a quadrotor.

The project uses quaternion representation for UAV orientation to overcome limitations of Euler-angle based approaches such as singularities and gimbal lock. The controller will generate required control torques to track desired roll, pitch, and yaw commands.

The developed system will be implemented and tested using MATLAB/Simulink and MuJoCo simulation environments.

## Objectives

- Model the rotational dynamics of a UAV.
- Implement quaternion-based attitude representation.
- Convert desired roll, pitch, and yaw commands into reference quaternions.
- Design and simulate a quaternion-based P² attitude controller.
- Analyze attitude tracking performance for different orientation commands.
- Validate the controller using Simulink and MuJoCo simulations.
- Extend the system towards UAV position and trajectory control.

## Methodology

1. Generate reference attitude using Euler angle to quaternion conversion.
2. Calculate quaternion error between desired and measured UAV orientation.
3. Use the P² controller to generate corrective torque commands.
4. Simulate UAV angular dynamics and quaternion evolution.
5. Visualize UAV attitude response using simulation environments.
6. Extend the work towards translation and trajectory generation using interpolation techniques.

## Expected Outcomes

- A working quaternion-based UAV attitude controller.
- Stable tracking of desired roll, pitch, and yaw commands.
- Simulation validation using MATLAB/Simulink and MuJoCo.
- Foundation for complete UAV trajectory control.

## Tools and Technologies

| Tool | Purpose |
|---|---|
| MATLAB/Simulink | UAV modelling and controller simulation |
| MuJoCo | Physics-based UAV simulation |
| Python | Simulation and controller implementation |
| GitHub | Project documentation and version control |

## Future Scope

Future extensions include translational dynamics modelling, position control, trajectory planning using LERP, and smooth attitude trajectory generation using SLERP.
