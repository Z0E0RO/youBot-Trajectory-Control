import numpy as np
import pandas as pd
import os
import modern_robotics as mr
import math
import matplotlib.pyplot as plt

# Chassis dimensions
l = 0.235
w = 0.15
# Whell radius
r = 0.0475

# Initializing the accumulated error to 0
Xerr_total = np.zeros(6)

# Max Velocity achieveable by wheels and joints
MAX_VELOCITY = 12.3

k = 1

# To keep the values of Error
Xerr_log = []

# Proportional Gain matrix
Kp = np.diag([10, 10, 12, 8, 8, 8])

# Integral Gain matrix
Ki = np.diag([0.02, 0.02, 0.03, 0.01, 0.01, 0.01])

# Getting the current working directory and creating path for the csv file.
current_directory = os.getcwd()
path = os.path.join(current_directory, "CapstoneProject.csv")
Xerr_path = os.path.join(current_directory, "Xerr.csv")

# Twist matrix in body frame
Blist = np.array([[0, 0, 1, 0, 0.0330, 0],
                  [0, -1, 0, -0.5076, 0, 0],
                  [0, -1, 0, -0.3526, 0, 0],
                  [0, -1, 0, -0.2176, 0, 0],
                  [0, 0, 1, 0, 0, 0]]).T

# F matrix to determine twist based on a wheel configuration
F = (r/4) * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)], [1, 1, 1, 1], [-1, 1, -1, 1]])

# Empty list to store all the configurations of chassis, wheel, joints and gripper
final_configuration = []

def NextStep(initial, controls, max_velocity, gripper_state, timestep=0.01):
    '''
    Calculate the next state of mobile manipulator using odometry and simple Euler step
    :param initial: A 12-vector representing the current configuration of mobile manipulator (3 variables
    for chassis configuration, 5 variables for the arm and 4 variables for the wheel)
    :param controls: A 9-vector representing arm and wheel speed
    :param max_velocity: Scalar giving the maximum velocity that can be acheived by arm and wheels
    :param gripper_state: Scalar quantity represting the state of gripper (0 for open and 1 for close)
    :param timestep: A time step delta t
    :return: Return the 13-vector representing the new configuration of the mobile manipulator
    '''

    # Getting [phi, x, y]
    q = np.array(initial[0:3])
    # Clipping the out of range values of velocities
    controls = np.clip(controls, -max_velocity, max_velocity)

    # Separating wheel and joint speed
    wheel_speed = np.array(controls[0:4])
    joint_speed = np.array(controls[4:])

    # Separating wheel and joint angles
    wheel_angles = np.array(initial[8:12])
    joint_angles = np.array(initial[3:8])

    # integrate joint & wheel motion
    new_joint_angles = joint_angles + joint_speed * timestep
    new_wheel_angles = wheel_angles + wheel_speed * timestep

    # Using odemetry to find the new configuration of chassis
    # wheel kinematics
    delta_phi = (new_wheel_angles - wheel_angles).reshape((4,1))
    Vb = F @ delta_phi / timestep    # (3,1)
    delta_qb = (Vb * timestep).flatten()     # (3,) now scalars

    # chassis update
    if abs(delta_qb[0]) < 1e-6:  # pure translation
        dq = np.array([0, delta_qb[1], delta_qb[2]])
    else:                        # rotation + translation
        dtheta = delta_qb[0]
        dq = np.array([
            dtheta,
            (delta_qb[1] * np.sin(dtheta) + delta_qb[2] * (1 - np.cos(dtheta))) / dtheta,
            (delta_qb[2] * np.sin(dtheta) + delta_qb[1] * (np.cos(dtheta) - 1)) / dtheta
        ])

    # map body twist to space frame
    phi = q[0]
    delta_q = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi),  np.cos(phi)]
    ]) @ dq

    q_new = q + delta_q
    new_config = list(q_new) + list(new_joint_angles) + list(new_wheel_angles) + [gripper_state]

    # Appending the new configuration to the configuration list
    final_configuration.append(new_config.copy())

    return new_config

def trajectory_time(X_start, X_end, linear_velocity=0.15, angular_velocity=0.15, dt=0.01):
    '''
    Function to calculate the trajectory time 
    :param X_start: Configuration of the starting point
    :param X_end: Configuration of the goal point
    :param linear_velocity: Maximum linear velocity for the end effector
    :param angular_velocity: Maximum angular velocity for the end effector
    :param dt: Sample time delta t
    :return: The time to complete that trajectory
    '''

    # Separating the rotation matrix and translation matrix
    R_start, p_start = mr.TransToRp(X_start)
    R_end, p_end = mr.TransToRp(X_end)

    # Calculating the linear and angular distances
    linear_distance = math.sqrt((p_end[0] - p_start[0])**2 + (p_end[1] - p_start[1])**2 + (p_end[2] - p_start[2])**2)
    angular_distance = np.arccos((np.trace(R_end) - 1) / 2) - np.arccos((np.trace(R_start) - 1) / 2)

    # Calculating the linear and angular time
    linear_time = linear_distance / linear_velocity
    angular_time = angular_distance / angular_velocity

    # Taking the maximum time
    Tf = max(linear_time, angular_time)
    # Making that time a multiple of sample time
    Tf  = math.ceil(Tf / dt) * dt
    return Tf

def trajectory_steps(X_start, X_end, k, gripper_state, dt=0.01):
    '''
    Function to generate the trajectory between two configurations
    :param X_start: Configuration of the starting point
    :param X_end: Configuration of the goal point
    :param k: The amount of step per time step
    :param gripper_state: The state of the gripper whether it is open(0) or close(1)
    :param dt: Sample time delta t
    :return: The trajectory between the start and goal
    '''

    # Getting the time for the trajectory
    Tf = trajectory_time(X_start, X_end)

    # Calculating the number of steps
    N = int(Tf/dt) * k

    # Checking if the start and goal configuration are the same doing this because gripper takes 0.625s
    # to completely open and close
    if N == 0:
        trajectory = [X_end for _ in range(math.ceil(0.625 / dt))]
    else:
        # Generate the Screw trajectory
        trajectory = mr.ScrewTrajectory(X_start, X_end, Tf, N, 5)
    return trajectory

def format_trajectory(trajectory, gripper_state):
    '''
    Function to format the complete trajector
    :param trajectory: The trajectory to include
    :param gripper_state: The state of the gripper
    :return: The trajectory formatted in the desired way
    '''

    # Empty list used to format the trajectory
    formatted = []

    # Iterating over all the configuration in the trajectory
    for T in trajectory:
        r11, r12, r13, px = T[0, :]
        r21, r22, r23, py = T[1, :]
        r31, r32, r33, pz = T[2, :]
        # Appending the values in the formatted list
        formatted.append([r11, r12, r13,
                          r21, r22, r23,
                          r31, r32, r33,
                          px, py, pz,
                          gripper_state])
    return formatted

def trajectory_generation(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k):
    '''
    Function to generate and store the complete trajectory of end effector between multiple points
    :param Tse_initial: The initial configuration of end effector relative to the space
    :param Tsc_initial: The initial configuration of cube relative to the space
    :param Tsc_final: The final configuration of cube relative to the space
    :param Tce_grasp: The end-effector configuration while grasping the cube relative to the cube frame
    :param Tce_standoff: The end-effector standoff configuration near the cube relative to the cube frame
    :return: Full trajectory of all the points
    '''
    
    full_trajectory = []

    # Step 1: Move from Tse initial to Tse standoff
    Tse_standoff_initial = Tsc_initial @ Tce_standoff
    s1 = trajectory_steps(Tse_initial, Tse_standoff_initial, k, 0)
    full_trajectory += format_trajectory(s1, 0)

    # Step 2: Move from Tse stand off to Tse grasp
    Tse_grasp_initial = Tsc_initial @ Tce_grasp
    s2 = trajectory_steps(Tse_standoff_initial, Tse_grasp_initial, k, 0)
    full_trajectory += format_trajectory(s2, 0)

    # Step 3: Stay at Tse grasp and grasp the block
    s3 = trajectory_steps(Tse_grasp_initial, Tse_grasp_initial, k, 1)
    full_trajectory += format_trajectory(s3, 1)


    # Step 4: Move from Tse grasp to Tse standoff
    s4 =  trajectory_steps(Tse_grasp_initial, Tse_standoff_initial, k, 1)
    full_trajectory += format_trajectory(s4, 1)

    # Step 5: Move from Tse standoff initial to Tse standoff final
    Tse_standoff_final = Tsc_final @ Tce_standoff
    s5 = trajectory_steps(Tse_standoff_initial, Tse_standoff_final, k, 1)
    full_trajectory += format_trajectory(s5, 1)

    # Step 6: Move from Tse standoff final to Tse grasp final
    Tse_grasp_final = Tsc_final @ Tce_grasp
    s6 = trajectory_steps(Tse_standoff_final, Tse_grasp_final, k, 1)
    full_trajectory += format_trajectory(s6, 1)

    # Step 7: Drop the block here
    s7 = trajectory_steps(Tse_grasp_final, Tse_grasp_final, k, 0)
    full_trajectory += format_trajectory(s7, 0)

    # Step 8: Move back from Tse grasp final to Tse standoff final
    s8 = trajectory_steps(Tse_grasp_final, Tse_standoff_final, k, 0)
    full_trajectory += format_trajectory(s8, 0)

    return full_trajectory

def FeedBackControl(X, Xd, Xd_next, Kp, Ki, Xerr_total, dt=0.01):
    '''
    Function to generate Twist (V)
    :param X: Current end-effector configuration
    :param Xd: Reference end-effector configuration
    :param Xd_next: Reference end-effector configuration at next time step
    :param Kp: Proportional Gain Matrix
    :param Ki: Integral Gain Matrix
    :param Xerr_total: The total accumulated error
    :param dt: Time step delta t
    :return: Twist and total accumulated error
    '''
    # Calculating Vd
    Vd_se3 = mr.MatrixLog6(mr.TransInv(Xd) @ Xd_next)
    Vd = (1.0 / dt) * mr.se3ToVec(Vd_se3)

    # Calculating Xerr (Error)
    Xerr_se3 = mr.MatrixLog6(mr.TransInv(X) @ Xd)
    Xerr = mr.se3ToVec(Xerr_se3)
    Xerr_log.append(Xerr)

    # Calculating Feedforward, Proportional and Integral terms
    FF_term = mr.Adjoint(mr.TransInv(X) @ Xd) @ Vd
    Proportional = Kp @ Xerr
    Xerr_total += Xerr * dt
    Integral = Ki @ Xerr_total
    V = FF_term + Proportional + Integral
    return V, Xerr_total

def Tsb_creation(phi, x, y):
    '''
    Function to calculate the Transformation matrix of chassis relative to the space frame
    :param phi: Angle representing the turning angle of chassis (in rad)
    :param x: Chassis translation in x-axis
    :param y: chassis traslation in y-axis
    :return: Transformation matrix of chassis relative to the space frame
    '''
    Tsb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                    [np.sin(phi), np.cos(phi), 0, y],
                    [0, 0, 1, 0.0963],
                    [0, 0, 0, 1]])
    return Tsb

def Tse_creation(Tsb, Tb0, M0e, configuration):
    '''
    Function to calculate end-effector configuration (transformation matrix) relative to space frame
    :param Tsb: Transformation matrix of chassis relative to space frame
    :param Tb0: Transformation matrix of arm base relative to chassis frame
    :param M0e: End-effector home configuration
    :param configuration: Current configuration of the mobile manipulator
    :return: end-effector configuration (transformation matrix) relative to space frame
    '''
    # Getting joint angles from the configuration
    thetalist = configuration[3:8]
    # calculating Tse by using Forward Body Kinematics
    Tse = Tsb @ Tb0 @ mr.FKinBody(M0e, Blist, thetalist)
    return Tse

def Jacobian(Blist, configuration, M0e, Tb0):
    '''
    Function to calculate Jacobian which maps joint and wheel velocities to end-effector
    :param Blist: Screw axis in body frame
    :param configuration: Current configuration of mobile manipulator
    :param M0e: End-effector home configuration
    :param Tb0: Transformation matrix of arm base relative to chassis frame
    :return: Jacobian (J_base and J_arm)
    '''
    # Get the link angles from the configuration
    thetalist = configuration[3:8]
    # Calculate end-effector configuration by using Forward body kinematics
    T0e = mr.FKinBody(M0e, Blist, thetalist)
    # Initialize the F6 matrix to have all entries 0
    F6 = np.zeros((6, 4))
    # Create F6 matrix = [0, 0, F, 0]
    F6[2:5, :] = F
    # Calculate J_base
    J_base = mr.Adjoint(mr.TransInv(T0e) @ mr.TransInv(Tb0)) @ F6
    # Calculate J_arm
    J_arm = mr.JacobianBody(Blist, thetalist)
    # Combine J_base and J_arm
    J = np.concatenate((J_base, J_arm), axis=1)
    return J

# Main Program to utlize above functions and create the csv file

# Initial configuration of mobile manipulator
initial_configuration = [-0.75959, -0.47352, 0.058167, 0.80405, -0.91639, -0.011436, 0.054333, 
                         0.00535, 1.506, -1.3338, 1.5582, 1.6136, 0]

#initial_configuration = [-0.15959, -0.17352, -0.041833, 0.80405, -0.91639, -0.011436, 0.054333, 0.00535, 1.506, -1.3338, 1.5582, 1.6136, 0] # Offest configuration

# Creating T_sb
T_sb = Tsb_creation(initial_configuration[0], initial_configuration[1], initial_configuration[2])

# Configuration of arm base relative to the chassis frame
T_b0 = np.array([[1, 0, 0, 0.1662],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0.0026],
                 [0, 0, 0, 1]])

# Arm home configuration
M_0e = np.array([[1, 0, 0, 0.033],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0.6546],
                 [0, 0, 0, 1]])


# Cube initial configuration relative to the space frame
Tsc_initial = np.array([[1, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.025],
                        [0, 0, 0, 1]])

# Cube final configuration relative to the space frame
Tsc_final = np.array([[0, 1, 0, 0],
                      [-1, 0, 0, -1],
                      [0, 0, 1, 0.025],
                      [0, 0, 0, 1]])


# End-effector configuration relative to the cube while grasping it

Tce_grasp = np.array([[0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])

# End-effector configuration relative to the cube in its standoff mode

Tce_standoff = np.array([[0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0.1],
                         [0, 0, 0, 1]])

# Creating the initial configuration of end-effector relative to the space frame

# Tse with offset error
Tse_initial = np.array([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0.5],
                        [0, 0, 0, 1]])


final_configuration.append(initial_configuration.copy())

X = Tse_creation(T_sb, T_b0, M_0e, initial_configuration)

# Generating the trajectory
trajectory = trajectory_generation(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k)

# Iterating over the trajectory
for i in range(len(trajectory)):
    # Getting end-effector configuration of reference trajectory
    Xd = [[trajectory[i][0], trajectory[i][1], trajectory[i][2], trajectory[i][9]],
          [trajectory[i][3], trajectory[i][4], trajectory[i][5], trajectory[i][10]],
          [trajectory[i][6], trajectory[i][7], trajectory[i][8], trajectory[i][11]],
          [0, 0, 0, 1]]
    
    # Getting the end-effector configuration for the next time step
    if i == len(trajectory) - 1:
        Xd_next = Xd
    else:
        Xd_next = [[trajectory[i+1][0], trajectory[i+1][1], trajectory[i+1][2], trajectory[i+1][9]],
                   [trajectory[i+1][3], trajectory[i+1][4], trajectory[i+1][5], trajectory[i+1][10]],
                   [trajectory[i+1][6], trajectory[i+1][7], trajectory[i+1][8], trajectory[i+1][11]],
                   [0, 0, 0, 1]]
    
    # Taking gripper_state during the trajectory
    gripper_state = trajectory[i][12]

    # Getting Twist(V) and total accumulated error from FeedBackControl function
    V, Xerr_total = FeedBackControl(X, Xd, Xd_next, Kp, Ki, Xerr_total)

    # Calculating the Jacobian
    Je = Jacobian(Blist, initial_configuration, M_0e, T_b0)
    W = np.diag([5, 5, 5, 5, 1, 1, 1, 1, 1]) # Favor chassis
    Je_weighted = W @ Je.T  # (9, 6)
    controls = np.linalg.pinv(Je_weighted @ Je) @ (Je_weighted @ V)
    # Separating the wheel and joint speeds
    wheel_controls = controls[:4]
    joint_controls = controls[4:]

    # Combine them properly
    combined_controls = np.concatenate((wheel_controls, joint_controls))

    # Calling NextStep function to get the next configuration
    initial_configuration = NextStep(initial_configuration, combined_controls, MAX_VELOCITY, gripper_state)
    T_sb = Tsb_creation(initial_configuration[0], initial_configuration[1], initial_configuration[2])
    Tse_initial = Tse_creation(T_sb, T_b0, M_0e, initial_configuration)
    X = Tse_initial
    # print("Wheel speeds:", controls[:4])
    # print("Joint speeds:", controls[4:])

Xerr_array = np.array(Xerr_log)

# Calculating the total time
times = np.arange(0, len(Xerr_log) * 0.01, 0.01)
labels = ['ωx', 'ωy', 'ωz', 'vx', 'vy', 'vz']
# Plotting Xerr vs Time graph
plt.figure()
for i in range(6):
    plt.plot(times, Xerr_array[:, i], label=labels[i])
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('End-Effector Error Over Time')
plt.legend()
plt.grid()
plt.savefig('Xerr_plot.pdf')
plt.show()

# Creating the csv files
pd.DataFrame(final_configuration).to_csv(path, header=False, index=False)
pd.DataFrame(Xerr_array).to_csv(Xerr_path, index=False, header=False)
