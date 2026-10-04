# 🤖 KUKA youBot Mobile Manipulation

Trajectory planning and feedback control for the **KUKA youBot mobile manipulator**, developed as part of the **Modern Robotics – Mobile Manipulation Capstone Project** by Northwestern University.

The project combines mobile-base motion and manipulator control to perform an autonomous **pick-and-place task** in simulation.

## Project Overview

The controller generates a complete end-effector trajectory for the youBot to:

1. Approach a cube
2. Move to the grasp configuration
3. Pick up the cube
4. Transport it to a new location
5. Release the cube
6. Move back to a standoff position

The implementation includes:

- Screw-based trajectory generation
- Mobile-base odometry
- Forward kinematics
- Combined base and arm Jacobian
- Feedforward + PI feedback control
- Wheel and joint velocity limits
- End-effector tracking error logging
- Pick-and-place simulation

## Technologies

**Python** • **NumPy** • **Modern Robotics** • **Matplotlib** • **Pandas**

## Control Architecture

The project implements the main components of mobile manipulation control:

`Reference Trajectory`

↓

`Feedforward + PI Feedback Controller`

↓

`Combined Base + Arm Jacobian`

↓

`Wheel & Joint Velocities`

↓

`Robot Configuration Update`

The controller continuously compares the desired and actual end-effector configurations and calculates the required motion of both the mecanum-wheel base and robotic arm.

## Project Structure

```text
youBot-Trajectory-Control/
│
├── src/
│   └── main.py
│
├── data/
│   ├── Project.csv
│   ├── Xerr.csv
│   └── Xerr_plot.pdf
│
├── simulation/
│   ├── Simulation Video.mp4
│   └── youBot_cube.ttt
│
├── requirements.txt
└── README.md
```

## Results

The controller successfully performs the complete simulated pick-and-place sequence while tracking the desired end-effector trajectory.

Tracking error for all six components of the end-effector twist is recorded during execution and stored in:

`data/Xerr.csv`

The generated error plot is available here:

[View End-Effector Tracking Error Plot](data/Xerr_plot.pdf)

## 🎥 Simulation

[▶ Watch the youBot Pick-and-Place Simulation](simulation/Simulation%20Video.mp4)

## Installation

Clone the repository:

```bash
git clone https://github.com/Z0E0RO/youBot-Trajectory-Control.git
cd youBot-Trajectory-Control
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the controller:

```bash
python src/main.py
```

## Background

This project was completed as part of the **Modern Robotics: Mechanics, Planning, and Control** specialization by **Northwestern University**.

It provided practical experience with robot kinematics, trajectory generation, mobile manipulation, Jacobian-based control, and feedback control.
