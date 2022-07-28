from typing import List, Tuple, Any
import numpy as np

import jax.scipy as jscipy
import jax.numpy as jnp
from jax import grad, jit, jacfwd, jacrev
from jax.config import config
config.update("jax_enable_x64", True)

def unit_vec_jnp(q, q_dot, q_ddot):
    """unit vector function provides three different normalization method/scale to the three three-element vectors"""

    collection = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    nq = jnp.linalg.norm(q)
    u = q / nq
    u_dot = q_dot / nq - q * jnp.dot(q, q_dot) / jnp.power(nq,3)

    u_ddot = q_ddot / nq - q_dot / jnp.power(nq,3) * (2 * jnp.dot(q, q_dot)) - q / jnp.power(nq,3) * (jnp.dot(q_dot, q_dot) + jnp.dot(q, q_ddot)) + 3 * q / pow(nq,5) * pow(jnp.dot(q, q_dot),2)
    # breakpoint()
    collection = collection.at[0:3].set(u)
    collection = collection.at[3:6].set(u_dot)
    collection = collection.at[6:9].set(u_ddot)

    # breakpoint()

    return collection

def hatOperator_jnp(v):
    hat = jnp.zeros((3, 3))
    # breakpoint()
    hat = hat.at[2,1].set(v[0])
    hat = hat.at[2,1].set(v[0])
    hat = hat.at[1,2].set(-v[0])
    hat = hat.at[0,2].set(v[1])
    hat = hat.at[2,0].set(-v[1])
    hat = hat.at[1,0].set(v[2])
    hat = hat.at[0,1].set(-v[2])
    return hat

def veeOperator_jnp(input):
    output = jnp.zeros((3,))
    output = output.at[0].set(input[2][1])
    output = output.at[1].set(input[0][2])
    output = output.at[2].set(input[1][0])

    return output

def geo_control_jnp(J, kg_vehicleMass, currentTime, state):
    # J = jnp.array([
    # [2.5*1e-3, 0, 0],
    # [0, 2.1*1e-3, 0],
    # [0, 0, 4.3*1e-3]
    # ])
    # kg_vehicleMass = 0.752
    # currentTime = 0

    GeoCtrl_Kpx = 16.*kg_vehicleMass # 4.512
    GeoCtrl_Kpy = 16.*kg_vehicleMass #4.512

    GeoCtrl_Kpz = 16.*kg_vehicleMass
    GeoCtrl_Kvx = 5.6*kg_vehicleMass
    GeoCtrl_Kvy = 5.6*kg_vehicleMass # 0.5
    GeoCtrl_Kvz = 5.6*kg_vehicleMass
    GeoCtrl_KRx = 8.81
    GeoCtrl_KRy = 8.81
    GeoCtrl_KRz = 8.81
    GeoCtrl_KOx = 2.54

    GeoCtrl_KOy = 2.54 # 0.073
    GeoCtrl_KOz = 2.54
    GRAVITY_MAGNITUDE = 9.81

    zeros2 = [0.0,0.0]
    zeros3 = [0.0,0.0,0.0]

    targetPos = jnp.array([2*(1-jnp.cos(currentTime)), 2*jnp.sin(currentTime), 1.0])

    targetVel = jnp.array([2*jnp.sin(currentTime), 2*jnp.cos(currentTime), 0])

    targetAcc = jnp.array([2*jnp.cos(currentTime), -2*jnp.sin(currentTime), 0])

    targetJerk = jnp.array([-2*jnp.sin(currentTime), -2*jnp.cos(currentTime), 0])

    targetSnap = jnp.array([-2*jnp.cos(currentTime), 2*jnp.sin(currentTime), 0])

    targetYaw = jnp.array([1.0,0.0]) # represent the orientation vector (Algo 1 in the supplementary)
    targetYaw_dot = jnp.array(zeros2) # represent derivative of the orientation vector
    targetYaw_ddot = jnp.array(zeros2)

    # begin geometric control
    # Position Error (ep)
    statePos = jnp.array([state[0],state[1],state[2]])
    r_error = statePos - targetPos

    # Velocity Error (ev)
    stateVel = jnp.array([state[3],state[4],state[5]])
    v_error = stateVel - targetVel

    target_force = jnp.array(zeros3)
    target_force = target_force.at[0].set(kg_vehicleMass * targetAcc[0] - GeoCtrl_Kpx * r_error[0] - GeoCtrl_Kvx * v_error[0])
    target_force = target_force.at[1].set(kg_vehicleMass * targetAcc[1] - GeoCtrl_Kpy * r_error[1] - GeoCtrl_Kvy * v_error[1])
    target_force = target_force.at[2].set(kg_vehicleMass * (targetAcc[2] - GRAVITY_MAGNITUDE) - GeoCtrl_Kpz * r_error[2] - GeoCtrl_Kvz * v_error[2])
    Rvec = jnp.array(state[6:15])
    R = jnp.reshape(Rvec, (3,3))

    z_axis = R[:,2]

    # target thrust [F] (z-positive)
    target_thrust = -jnp.dot(target_force,z_axis)
    # Calculate axis [zB_des] (z-positive)
    z_axis_desired = -target_force/jnp.linalg.norm(target_force)

    # [xC_des]
    x_c_des = jnp.array(zeros3)
    x_c_des = x_c_des.at[0].set(targetYaw[0])
    x_c_des = x_c_des.at[1].set(targetYaw[1])
    x_c_des = x_c_des.at[2].set(0)

    x_c_des_dot = jnp.array(zeros3)
    x_c_des_dot = x_c_des_dot.at[0].set(targetYaw_dot[0])
    x_c_des_dot = x_c_des_dot.at[1].set(targetYaw_dot[1])
    x_c_des_dot = x_c_des_dot.at[2].set(0)

    x_c_des_ddot = jnp.array(zeros3)
    x_c_des_ddot = x_c_des_ddot.at[0].set(targetYaw_ddot[0])
    x_c_des_ddot = x_c_des_ddot.at[1].set(targetYaw_ddot[1])
    x_c_des_ddot = x_c_des_ddot.at[2].set(0)
    # [yB_des]
    y_axis_desired = jnp.cross(z_axis_desired, x_c_des)
    y_axis_desired = y_axis_desired/jnp.linalg.norm(y_axis_desired)

    # [xB_des]
    x_axis_desired = jnp.cross(y_axis_desired, z_axis_desired)

    # [eR]
    # Slow version
    Rdes = jnp.empty(shape=(3,3))
    Rdes = Rdes.at[:,0].set(x_axis_desired)
    Rdes = Rdes.at[:,1].set(y_axis_desired)
    Rdes = Rdes.at[:,2].set(z_axis_desired)
    
    eRM = (jnp.matmul(jnp.transpose(Rdes),R) - jnp.matmul(jnp.transpose(R), Rdes)) / 2

    # Matrix3<T>(const T ax, const T ay, const T az,
    #  const T bx, const T by, const T bz,
    #  const T cx, const T cy, const T cz)
    # eR.x = eRM.c.y;
    # eR.y = eRM.a.z;
    # eR.z = eRM.b.x;
    eR = jnp.array(zeros3)
    eR = veeOperator_jnp(eRM)
    # print(eR)
    # eR[0] = eRM[2,1]
    # eR[1] = eRM[0,2]
    # eR[2] = eRM[1,0]

    Omega = jnp.array([state[15], state[16], state[17]])
    # print(Omega)

    #compute Omegad
    a_error = jnp.array(zeros3) # error on acceleration
    # a_error = [0,0,-GRAVITY_MAGNITUDE] + R[:,2]* target_thrust / kg_vehicleMass - targetAcc
    a_error = jnp.array([0,0,GRAVITY_MAGNITUDE]) - R[:,2]* target_thrust / kg_vehicleMass - targetAcc
    # ? turn GRAVITY_MAGNITUDE to - GRAVITY_MAGNITUDE
    # ? turn - R[:,2]* target_thrust / kg_vehicleMass to + R[:,2]* target_thrust / kg_vehicleMass

    target_force_dot = jnp.array(zeros3) # derivative of target_force
    target_force_dot = target_force_dot.at[0].set(- GeoCtrl_Kpx * v_error[0] - GeoCtrl_Kvx * a_error[0] + kg_vehicleMass * targetJerk[0])
    target_force_dot = target_force_dot.at[1].set(- GeoCtrl_Kpy * v_error[1] - GeoCtrl_Kvy * a_error[1] + kg_vehicleMass * targetJerk[1])
    target_force_dot = target_force_dot.at[2].set(- GeoCtrl_Kpz * v_error[2] - GeoCtrl_Kvz * a_error[2] + kg_vehicleMass * targetJerk[2])

    b3_dot = jnp.matmul(jnp.matmul(R, hatOperator_jnp(Omega)),jnp.array([0,0,1])) #derivative of (Re3) in eq (2)
    target_thrust_dot = - jnp.dot(target_force_dot,R[:,2]) - jnp.dot(target_force, b3_dot)
    # target_thrust_dot = + np.dot(target_force_dot,R[:,2]) + np.dot(target_force, b3_dot)
    # ? turn the RHS from - to +

    j_error = jnp.array(zeros3) # error on jerk
    # j_error = np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass + b3_dot * target_thrust / kg_vehicleMass - targetJerk
    j_error = -jnp.dot(R[:,2], target_thrust_dot) / kg_vehicleMass - b3_dot * target_thrust / kg_vehicleMass - targetJerk
    # ? turn - np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass to np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass
    # ? turn - b3_dot * target_thrust / kg_vehicleMass to + b3_dot * target_thrust / kg_vehicleMass

    target_force_ddot = jnp.array(zeros3) # derivative of target_force_dot
    target_force_ddot = target_force_ddot.at[0].set(- GeoCtrl_Kpx * a_error[0] - GeoCtrl_Kvx * j_error[0] + kg_vehicleMass * targetSnap[0])
    target_force_ddot = target_force_ddot.at[1].set(- GeoCtrl_Kpy * a_error[1] - GeoCtrl_Kvy * j_error[1] + kg_vehicleMass * targetSnap[1])
    target_force_ddot = target_force_ddot.at[2].set(- GeoCtrl_Kpz * a_error[2] - GeoCtrl_Kvz * j_error[2] + kg_vehicleMass * targetSnap[2])


    b3cCollection = jnp.array([zeros3,zeros3,zeros3]) # collection of three three-dimensional vectors b3c, b3c_dot, b3c_ddot
    b3cCollection = unit_vec_jnp(-target_force, target_force_dot, -target_force_ddot) # unit_vec_jnp function is from geometric controller's git repo: https://github.com/fdcl-gwu/uav_geometric_control/blob/master/matlab/aux_functions/deriv_unit_vec_jnptor.m
    
    b3c = jnp.array(zeros3)
    b3c_dot = jnp.array(zeros3)
    b3c_ddot = jnp.array(zeros3)

    b3c = b3c.at[0].set(b3cCollection[0])
    b3c = b3c.at[1].set(b3cCollection[1])
    b3c = b3c.at[2].set(b3cCollection[2])

    b3c_dot = b3c_dot.at[0].set(b3cCollection[3])
    b3c_dot = b3c_dot.at[1].set(b3cCollection[4])
    b3c_dot = b3c_dot.at[2].set(b3cCollection[5])

    b3c_ddot = b3c_ddot.at[0].set(b3cCollection[6])
    b3c_ddot = b3c_ddot.at[1].set(b3cCollection[7])
    b3c_ddot = b3c_ddot.at[2].set(b3cCollection[8])

    """some changes start here"""
    A2 = - jnp.matmul(hatOperator_jnp(x_c_des), b3c)
    A2_dot = - jnp.matmul(hatOperator_jnp(x_c_des_dot),b3c) - jnp.matmul(hatOperator_jnp(x_c_des), b3c_dot)
    A2_ddot = - jnp.matmul(hatOperator_jnp(x_c_des_ddot), b3c) - jnp.matmul(hatOperator_jnp(x_c_des_dot), b3c_dot) * 2 - jnp.matmul(hatOperator_jnp(x_c_des), b3c_ddot)

    b2cCollection = jnp.array([zeros3,zeros3,zeros3]) # collection of three three-dimensional vectors b2c, b2c_dot, b2c_ddot
    b2cCollection = unit_vec_jnp(A2, -A2_dot, A2_ddot) # unit_vec_jnp function is from geometric controller's git repo: https://github.com/fdcl-gwu/uav_geometric_control/blob/master/matlab/aux_functions/deriv_unit_vec_jnptor.m

    b2c = b2cCollection[0:3]
    b2c_dot = b2cCollection[3:6]
    b2c_ddot = b2cCollection[6:9]

    b1c_dot = jnp.matmul(hatOperator_jnp(b2c_dot), b3c) + jnp.matmul(hatOperator_jnp(b2c), b3c_dot)
    b1c_ddot = jnp.matmul(hatOperator_jnp(b2c_ddot),b3c) + jnp.matmul(hatOperator_jnp(b2c_dot), b3c_dot) * 2 + jnp.matmul(hatOperator_jnp(b2c), b3c_ddot)

    Rd_dot = jnp.empty(shape=(3,3)) # derivative of Rdes
    Rd_ddot = jnp.empty(shape=(3,3)) # derivative of Rd_dot

    Rd_dot = Rd_dot.at[0,:].set(b1c_dot)
    Rd_dot = Rd_dot.at[1,:].set(b2c_dot)
    Rd_dot = Rd_dot.at[2,:].set(b3c_dot)
    # Rd_dot = jnp.transpose(Rd_dot)

    Rd_ddot = Rd_ddot.at[0,:].set(b1c_ddot)
    Rd_ddot = Rd_ddot.at[1,:].set(b2c_ddot)
    Rd_ddot = Rd_ddot.at[2,:].set(b3c_ddot)
    # Rd_ddot = jnp.transpose(Rd_ddot)

    Omegad = veeOperator_jnp(jnp.matmul(jnp.transpose(Rdes), Rd_dot))
    # print(currentTime)
    # print(Omegad)
    Omegad_dot = veeOperator_jnp(jnp.matmul(jnp.transpose(Rdes), Rd_ddot) - jnp.matmul(hatOperator_jnp(Omegad), hatOperator_jnp(Omegad)))

    # these two lines are remedy which is not supposed to exist in the code. There might be an error in the code above.
    # Omegad[1] = -Omegad[1]
    # Omegad_dot[1] = -Omegad_dot[1]
    # temporarily use zero Omegad
    ew = Omega - jnp.matmul(jnp.matmul(jnp.transpose(R), Rdes), Omegad)
    # Moment: simple version
    M = jnp.array(zeros3)
    M = M.at[0].set(-GeoCtrl_KRx * eR[0] - GeoCtrl_KOx * ew[0])
    M = M.at[1].set(-GeoCtrl_KRy * eR[1] - GeoCtrl_KOy * ew[1])
    M = M.at[2].set(-GeoCtrl_KRz * eR[2] - GeoCtrl_KOz * ew[2])
    # Moment: full version
    M = M - jnp.matmul(J, (jnp.matmul(hatOperator_jnp(Omega), jnp.matmul(R.transpose(),jnp.matmul(Rdes, Omegad))) - jnp.matmul(jnp.transpose(R), jnp.matmul(Rdes, Omegad_dot))))
    # ShengC: an additive term is the following
    momentAdd = jnp.cross(Omega, (jnp.matmul(J, Omega))) # J is the inertia matrix
    M = M +  momentAdd

    thrustMomentCmd = jnp.array([0.0,0.0,0.0,0.0])
    thrustMomentCmd = thrustMomentCmd.at[0].set(target_thrust)
    thrustMomentCmd = thrustMomentCmd.at[1].set(M[0])
    thrustMomentCmd = thrustMomentCmd.at[2].set(M[1])
    thrustMomentCmd = thrustMomentCmd.at[3].set(M[2])

    # u = np.array([0.0,0.0,0.0,0.0])
    # motorAssignMatrix = np.array([[1, 1, 1, 1],
    #                               [-0.1, 0.1,-0.1, 0.1],
    #                               [-0.075, 0.075, 0.075, -0.075],
    #                               [-0.022, -0.022, 0.022, 0.022]])
    # u = np.matmul(LA.inv(motorAssignMatrix),thrustMomentCmd) # no need to re-assign to every motor in simulation
    u = thrustMomentCmd
    return u

def L1AC_jnp(state, ctrlcmd, din, L1_params):
    (As_v, As_omega, dt, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, kg_vehicleMass, GRAVITY_MAGNITUDE, J ) = L1_params
    # (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev, 
    # u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev, 
    # lpf1_prev, lpf2_prev) = din
    v_hat_prev = din[:3]
    omega_hat_prev = din[3:6]
    R_prev = jnp.reshape(din[6:15],(3,3))
    v_prev = din[15:18]
    omega_prev = din[18:21]

    u_b_prev = din[21:25]
    u_ad_prev = din[25:29]
    sigma_m_hat_prev = din[29:33]
    sigma_um_hat_prev = din[33:35]
    lpf1_prev = din[35:39]
    lpf2_prev = din[39:43]

    thrustMomentCmd = ctrlcmd
    # == begin L1 adaptive control ==
    # first do the state predictor
    e3 = jnp.array([0.0, 0.0, 1.0])
    # load translational velocity
    v_now = state[3:6]

    # load rotational velocity
    omega_now = state[15:18]

    massInverse = 1.0 / kg_vehicleMass

    # compute prediction error (on previous step)
    vpred_error_prev = v_hat_prev - v_prev # computes v_tilde for (k-1) step
    omegapred_error_prev = omega_hat_prev - omega_prev # computes omega_tilde for (k-1) step

    v_hat = v_hat_prev + (e3 * GRAVITY_MAGNITUDE - R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0]) * massInverse + R_prev[:,0] * sigma_um_hat_prev[0] * massInverse + R_prev[:,1] * sigma_um_hat_prev[1] * massInverse + vpred_error_prev * As_v) * dt
    Jinv = jnp.linalg.inv(J)
    # temp vector: thrustMomentCmd[1--3] + u_ad_prev[1--3] + sigma_m_hat_prev[1--3]
    # original form
    tempVec = jnp.array([u_b_prev[1] + u_ad_prev[1] + sigma_m_hat_prev[1], u_b_prev[2] + u_ad_prev[2] + sigma_m_hat_prev[2], u_b_prev[3] + u_ad_prev[3] + sigma_m_hat_prev[3]])
    omega_hat = omega_hat_prev + (-jnp.matmul(Jinv, jnp.cross(omega_prev, (jnp.matmul(J, omega_prev)))) + jnp.matmul(Jinv, tempVec) + omegapred_error_prev * As_omega) * dt

    # update the state prediction storage
    v_hat_prev = v_hat
    omega_hat_prev = omega_hat

    # compute prediction error (for this step)
    vpred_error = v_hat - v_now
    omegapred_error = omega_hat - omega_now

    # exponential coefficients coefficient for As
    exp_As_v_dt = jnp.exp(As_v * dt)
    exp_As_omega_dt = jnp.exp(As_omega * dt)

     # latter part of uncertainty estimation (piecewise constant) (step2: adaptation law)
    PhiInvmu_v = vpred_error / (exp_As_v_dt - 1) * As_v * exp_As_v_dt
    PhiInvmu_omega = omegapred_error / (exp_As_omega_dt - 1) * As_omega * exp_As_omega_dt

    sigma_m_hat = jnp.array([0.0,0.0,0.0,0.0]) # estimated matched uncertainty
    sigma_m_hat_2to4 = jnp.array([0.0,0.0,0.0]) # second to fourth element of the estimated matched uncertainty
    sigma_um_hat = jnp.array([0.0,0.0]) # estimated unmatched uncertainty

    # use the rotation matrix in the current step
    R = jnp.reshape(state[6:15], (3,3))

    sigma_m_hat = sigma_m_hat.at[0].set(jnp.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass)
    # turn np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass to -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
    sigma_m_hat_2to4 = -jnp.matmul(J, PhiInvmu_omega)
    sigma_m_hat = sigma_m_hat.at[1].set(sigma_m_hat_2to4[0])
    sigma_m_hat = sigma_m_hat.at[2].set(sigma_m_hat_2to4[1])
    sigma_m_hat = sigma_m_hat.at[3].set(sigma_m_hat_2to4[2])

    sigma_um_hat = sigma_um_hat.at[0].set(-jnp.dot(R[:,0], PhiInvmu_v) * kg_vehicleMass)
    sigma_um_hat = sigma_um_hat.at[1].set(-jnp.dot(R[:,1], PhiInvmu_v) * kg_vehicleMass)

    # store uncertainty estimations
    sigma_m_hat_prev = sigma_m_hat
    sigma_um_hat_prev = sigma_um_hat

    # compute lpf1 coefficients
    lpf1_coefficientThrust1 = jnp.exp(- ctoffq1Thrust * dt)
    lpf1_coefficientThrust2 = 1.0 - lpf1_coefficientThrust1

    lpf1_coefficientMoment1 = jnp.exp(- ctoffq1Moment * dt)
    lpf1_coefficientMoment2 = 1.0 - lpf1_coefficientMoment1

    # update the adaptive control
    u_ad_int = jnp.array([0.0,0.0,0.0,0.0])
    u_ad = jnp.array([0.0,0.0,0.0,0.0])

    # low-pass filter 1 (negation is added to u_ad_prev to filter the correct signal)
    u_ad_int = u_ad_int.at[0].set(lpf1_coefficientThrust1 * (lpf1_prev[0]) + lpf1_coefficientThrust2 * sigma_m_hat[0])
    u_ad_int = u_ad_int.at[1:3].set(lpf1_coefficientMoment1 * (lpf1_prev[1:3]) + lpf1_coefficientMoment2 * sigma_m_hat[1:3])

    lpf1_prev = u_ad_int # store the current state

    # coefficients for the second LPF on the moment channel
    lpf2_coefficientMoment1 = jnp.exp(- ctoffq2Moment * dt)
    lpf2_coefficientMoment2 = 1.0 - lpf2_coefficientMoment1

    # low-pass filter 2 (optional)
    u_ad = u_ad.at[0].set(u_ad_int[0]) # only one filter on the thrust channel
    u_ad = u_ad.at[1:3].set(lpf2_coefficientMoment1 * lpf2_prev[1:3] + lpf2_coefficientMoment2 * u_ad_int[1:3])

    lpf2_prev = u_ad # store the current state

    u_ad = -u_ad

    # store the values for next iteration (negation is added to u_ad_prev to filter the correct signal)
    u_ad_prev = u_ad

    v_prev = v_now
    omega_prev = omega_now
    R_prev = jnp.ravel(R)
    u_b_prev = thrustMomentCmd

    controlcmd_L1 = thrustMomentCmd + u_ad_prev

    din = jnp.concatenate((v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev, 
    u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev, 
    lpf1_prev, lpf2_prev), axis=0)
    return controlcmd_L1, din

def f_value_jnp(J, m, state,u):
    #Dynamics
    # state: [p(3), v(3), R_Matrix(9), omega(3)]
    ## p(3)-- x,y,z are position of drone in world/inertial frame
    ## v(3)-- vx,vy,vz are positional/linear velocities in world/inertial frame

    ## omega(3)-- rotational angular velocity in the body frame (from IMU)

    # control: [f, moments(3)]
    ## moments(3)-- the moments in the body frame
    g = 9.81
    vx = state[3]
    vy = state[4]
    vz = state[5]
    Rvec = state[6:15]
    Omega = state[15:18]  # angular velocity vector in body frame

    f_thrust = u[0]
    M = u[1:4]

    r_mat = jnp.reshape(Rvec,(3,3))
    e3 = jnp.array([0,0,1])

    thrust_proj = f_thrust/m*jnp.matmul(r_mat,e3)

    Rdot = jnp.matmul(r_mat,hatOperator_jnp(Omega))
    # # angular velocity in the inertial frame
    # Thetadot = omega2thetadot(Omega, Theta)
    # angular acceleration in body frame
    Omega_dot = jnp.matmul(jnp.linalg.inv(J),M - jnp.cross(Omega, jnp.matmul(J, Omega)))
    # print(Omega_dot)

    diff_state = jnp.array([vx,
              vy,
              vz,
              -thrust_proj[0],
              -thrust_proj[1],
              g - thrust_proj[2],
              Omega_dot[0],
              Omega_dot[1],
              Omega_dot[2]])
    return diff_state

def dynamicsL1_jac(combined_state: List[float], args: Tuple[Any]) -> List[float]:
    J = jacfwd(dynamicsL1, argnums=0)(combined_state, args)
    return jnp.asarray(J)

def dynamicsL1(combined_state: List[float], args: Tuple[Any]) -> float:
    state = jnp.array(combined_state[:18])
    din = jnp.array(combined_state[18:])
    t, dt, L1_params = args
    As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J = L1_params

    x = jnp.array([state[0], state[1], state[2]])
    xdot = jnp.array([state[3], state[4], state[5]])
    Rvec = jnp.array(state[6:15])
    Rmat = jnp.reshape(Rvec,(3,3))

    Omega = jnp.array([state[15], state[16], state[17]])

    controlcmd_geo = geo_control_jnp(J, m, t, state)
    controlcmd_L1, din_plus = L1AC_jnp(state, controlcmd_geo, din, L1_params)
    diff_state = f_value_jnp(J, m, state, controlcmd_L1)

    Rmat = jnp.matmul(Rmat, jscipy.linalg.expm(hatOperator_jnp(Omega)*dt))
    Omega_plus = Omega + dt*diff_state[6:9]
    xdot_plus = xdot + dt*diff_state[3:6]
    x_plus = x + dt*diff_state[0:3]
    Rvec_plus = jnp.ravel(Rmat)
    state_plus = jnp.concatenate((x_plus, xdot_plus, Rvec_plus, Omega_plus), axis=0)
    res_plus = jnp.concatenate((state_plus, din_plus), axis=0)
    # res_plus_array = jnp.asarray(res_plus)
    return res_plus

def dynamicsgeo_jac(combined_state: List[float], args: Tuple[Any]) -> List[float]:
    J = jacfwd(dynamicsgeo, argnums=0)(combined_state, args)
    return jnp.asarray(J)

def dynamicsgeo(combined_state: List[float], args: Tuple[Any]) -> float:
    state = jnp.array(combined_state[:18])
    # din = jnp.array(combined_state[18:])
    t, dt, L1_params = args
    As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J = L1_params

    x = jnp.array([state[0], state[1], state[2]])
    xdot = jnp.array([state[3], state[4], state[5]])
    Rvec = jnp.array(state[6:15])
    Rmat = jnp.reshape(Rvec,(3,3))

    Omega = jnp.array([state[15], state[16], state[17]])

    controlcmd_geo = geo_control_jnp(J, m, t, state)
    # controlcmd_L1, din_plus = L1AC_jnp(state, controlcmd_geo, din, L1_params)
    diff_state = f_value_jnp(J, m, state, controlcmd_geo)

    Rmat = jnp.matmul(Rmat, jscipy.linalg.expm(hatOperator_jnp(Omega)*dt))
    Omega_plus = Omega + dt*diff_state[6:9]
    xdot_plus = xdot + dt*diff_state[3:6]
    x_plus = x + dt*diff_state[0:3]
    Rvec_plus = jnp.ravel(Rmat)
    state_plus = jnp.concatenate((x_plus, xdot_plus, Rvec_plus, Omega_plus), axis=0)
    # res_plus = jnp.concatenate((state_plus, din_plus), axis=0)
    # res_plus_array = jnp.asarray(res_plus)
    return state_plus

def simulate(init, time_bound, time_step):
    J = 1e-3*jnp.diag(jnp.array([2.5, 2.1, 4.3]))
    m = 0.752
    g = 9.81
    N_step = int((time_bound)/time_step)
    state_init = init[0:18]
    din_init = init[18:]

    As_v = -5.0
    As_omega = -5.0
    dt_L1 = time_step

    ctoffq1Thrust = 5*7
    ctoffq1Moment = 1*7
    ctoffq2Moment = 1*7

    L1_params = (As_v, As_omega, dt_L1, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, m, g, J)

    traj: jnp.ndarray = jnp.array([[0] + state_init + din_init])
    dynamicsL1_jit = jit(dynamicsL1)
    dynamicsL1_jac_jit = jit(dynamicsL1_jac)
    for i in range(N_step):
        print(i)
        t = traj[-1,0]
        state = traj[-1,1:]
        state_plus = dynamicsL1_jit(state, (t, time_step, L1_params,))
        J = dynamicsL1_jac_jit(state, (t, time_step, L1_params,))
        t = t+time_step
        state_plus = jnp.concatenate((jnp.array([t]), state_plus), axis=0)
        traj = jnp.concatenate((traj, jnp.reshape(state_plus, (1,62))), axis=0)
    return traj

def dynamicsL1_jac_1d(combined_state, args, idx):
    res = jit(dynamicsL1_jac)(combined_state, args)
    return np.array(res[idx,:].astype('float64'))

def dynamicsL1_1d(combined_state, args, idx):
    res = jit(dynamicsL1)(combined_state, args)
    return res[idx].astype('float64')

def dynamicsgeo_jac_1d(combined_state, args, idx):
    res = jit(dynamicsgeo_jac)(combined_state, args)
    return np.array(res[idx,:].astype('float64'))

def dynamicsgeo_1d(combined_state, args, idx):
    res = jit(dynamicsgeo)(combined_state, args)
    return res[idx].astype('float64')

def dynamicsgeounroll(combined_state, args):
    t, time_step, L1_params, step_bound = args

    res = combined_state
    for i in range(step_bound):
        new_t = i*time_step
        new_args = (new_t, time_step, L1_params)
        res = jit(dynamicsgeo)(combined_state, new_args)
        combined_state = res
    return res

def dynamicsgeounroll_jac(combined_state, args):
    J = jacfwd(dynamicsgeounroll, argnums=0)(combined_state, args)
    return jnp.asarray(J)

def dynamicsgeounroll_1d(combined_state, args, idx):
    res = dynamicsgeounroll(combined_state, args)
    return res[idx].astype('float64')

def dynamicsgeounroll_jac_1d(combined_state, args, idx):
    res = dynamicsgeounroll_jac(combined_state, args)
    return np.array(res[idx,:].astype('float64'))

if __name__ == "__main__":
    # test()
    state_init = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]
    din_init = state_init[3:6]+state_init[15:18]+state_init[6:15]+\
        [0.0,0.0,0.0]+[0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]+\
        [0.0,0.0,0.0,0.0]+[0.0,0.0]+[0.0,0.0,0.0,0.0]+[0.0,0.0,0.0,0.0]
    init = state_init + din_init
    trace = simulate(init, 0.01, 0.001)
    import matplotlib.pyplot as plt
    import numpy as np 
    trace = np.array(trace)
    plt.plot(trace[:,1], trace[:,2])
    plt.show()
