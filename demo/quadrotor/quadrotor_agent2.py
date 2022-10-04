# Example agent.
from typing import Tuple, List
import json
import os
import numpy as np
from scipy.integrate import ode
import torch
import math
from verse.agents import BaseAgent
from verse.map import LaneMap
from verse.map.lane_map_3d import LaneMap_3d
import scipy
from scipy.integrate import odeint

class FFNNC(torch.nn.Module):
    def __init__(self, D_in=6, D_out=8):
        super(FFNNC, self).__init__()
        self.layer1 = torch.nn.Linear(D_in, 20)
        self.layer2 = torch.nn.Linear(20, 20)
        self.layer3 = torch.nn.Linear(20, D_out)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91
A = np.zeros([10, 10])
A[0, 1] = 1.
A[1, 2] = g
A[2, 2] = -d1
A[2, 3] = 1
A[3, 2] = -d0
A[4, 5] = 1.
A[5, 6] = g
A[6, 6] = -d1
A[6, 7] = 1
A[7, 6] = -d0
A[8, 9] = 1.

B = np.zeros([10, 3])
B[3, 0] = n0
B[7, 1] = n0
B[9, 2] = kT


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.
    dx/dt = A x + B u
    cost = integral x.T*Q*x + u.T*R*u
    """
    # http://www.mwm.im/lqr-controllers-with-python/
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return np.asarray(K), np.asarray(X), np.asarray(eigVals)

####################### solve LQR #######################
n = A.shape[0]
m = B.shape[1]
Q = np.eye(n)
Q[0, 0] = 10.
Q[1, 1] = 10.
Q[2, 2] = 10.
R = np.diag([1., 1., 1.])
K, _, _ = lqr(A, B, Q, R)

####################### The controller ######################
def u(x, goal):
    goal = np.array(goal)
    return K.dot([goal[0],0,0,0, goal[1],0,0,0, goal[2],0] - x) + [0, 0, g / kT]

# non-linear dynamics
def dynamics(x, u):
    x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz = x.reshape(-1).tolist()
    ax, ay, az = u.reshape(-1).tolist()
    dot_x = np.array([
    vx,
    g * np.tan(theta_x),
    -d1 * theta_x + omega_x,
    -d0 * theta_x + n0 * ax,
    vy,
    g * np.tan(theta_y),
    -d1 * theta_y + omega_y,
    -d0 * theta_y + n0 * ay,
    vz,
    kT * az - g])
    return dot_x


class QuadrotorAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)
        pass

    @staticmethod
    def cl_nonlinear(x, t, goal):
        x = np.array(x)
        dot_x = dynamics(x, u(x, goal))
        return dot_x


    def action_handler(self, mode, state, lane_map: LaneMap_3d):
        # if mode[0] == 'Normal':
        df = 0
        if lane_map.check_guard_box(self.id, state[3:9]):
            # lane_map.get_next_point(mode[1], self.id, state[3:6])
            df = 1
        # else:
        #     raise ValueError
        return df

    # def runModel(self, mode,  initalCondition, time_bound, time_step, ref_input, lane_map: LaneMap_3d):
    #     path = os.path.abspath(__file__)
    #     path = path.replace('quadrotor_agent.py', 'prarm.json')
    #     # print(path)
    #     with open(path, 'r') as f:
    #         prarms = json.load(f)
    #     bias1 = prarms['bias1']
    #     bias2 = prarms['bias2']
    #     bias3 = prarms['bias3']
    #     weight1 = prarms['weight1']
    #     weight2 = prarms['weight2']
    #     weight3 = prarms['weight3']

    #     bias1 = torch.FloatTensor(bias1)
    #     bias2 = torch.FloatTensor(bias2)
    #     bias3 = torch.FloatTensor(bias3)
    #     weight1 = torch.FloatTensor(weight1)
    #     weight2 = torch.FloatTensor(weight2)
    #     weight3 = torch.FloatTensor(weight3)
    #     controller = FFNNC()
    #     controller.layer1.weight = torch.nn.Parameter(weight1)
    #     controller.layer2.weight = torch.nn.Parameter(weight2)
    #     controller.layer3.weight = torch.nn.Parameter(weight3)
    #     controller.layer1.bias = torch.nn.Parameter(bias1)
    #     controller.layer2.bias = torch.nn.Parameter(bias2)
    #     controller.layer3.bias = torch.nn.Parameter(bias3)
    #     control_input_list = [[-0.1, -0.1, 7.81],
    #                           [-0.1, -0.1, 11.81],
    #                           [-0.1, 0.1, 7.81],
    #                           [-0.1, 0.1, 11.81],
    #                           [0.1, -0.1, 7.81],
    #                           [0.1, -0.1, 11.81],
    #                           [0.1, 0.1, 7.81],
    #                           [0.1, 0.1, 11.81]]
    #     init = initalCondition
    #     trajectory = [init]
    #     r = ode(self.dynamic)
    #     # r.set_initial_value(init)
    #     ex_list = []
    #     ey_list = []
    #     ez_list = []
    #     t = 0
    #     time = [t]
    #     trace = [[t]]
    #     trace[0].extend(init[3:])
    #     i = 0
    #     df = 0
    #     while (t < time_bound) and df == 0:
    #         ex = trajectory[i][3] - trajectory[i][0]
    #         ey = trajectory[i][4] - trajectory[i][1]
    #         ez = trajectory[i][5] - trajectory[i][2]
    #         # if self.id == 'test1':
    #         #     print([t, ex, ey, ez, trajectory[i][0],
    #         #           trajectory[i][1], trajectory[i][2]])
    #         evx = trajectory[i][6] - ref_input[0]
    #         evy = trajectory[i][7] - ref_input[1]
    #         evz = trajectory[i][8] - ref_input[2]

    #         sc = ref_input[3]  # math.atan2(dot, det)

    #         tmp1 = ex * math.cos(sc) - ey * math.sin(sc)
    #         tmp2 = ex * math.sin(sc) + ey * math.cos(sc)
    #         ex = tmp1
    #         ey = tmp2

    #         tmp1 = evx * math.cos(sc) - evy * math.sin(sc)
    #         tmp2 = evx * math.sin(sc) + evy * math.cos(sc)
    #         evx = tmp1
    #         evy = tmp2

    #         data = torch.FloatTensor(
    #             [0.2 * ex, 0.2 * ey, 0.2 * ez, 0.1 * evx, 0.1 * evy, 0.1 * evz])
    #         res = controller(data)
    #         res = res.detach().numpy()
    #         idx = np.argmax(res)
    #         u = control_input_list[idx] + ref_input[0:3] + [sc]

    #         df = self.action_handler(mode, init, lane_map)

    #         u = u+[df]
    #         init = trajectory[i]  # len 9
    #         r = ode(self.dynamic)
    #         r.set_initial_value(init)
    #         r.set_f_params(u)
    #         val = r.integrate(r.t + time_step)

    #         t = t+time_step
    #         if round(t-time_bound-time_step, 4) >= 0:
    #             break
    #         i += 1
    #         #  print(i,idx,u,res)
    #         trajectory.append(val)
    #         time.append(t)

    #         ex_list.append(ex)
    #         ey_list.append(ey)
    #         ez_list.append(ez)
    #         trace.append([t])
    #         # remove the reference trajectory from the trace
    #         trace[i].extend(val[3:])
    #         # if self.id == 'test1':
    #         #     print([t, trajectory[i][0], trajectory[i][1], trajectory[i][2]])
    #         #     print('curr', val[3:6])
    #     # print(trajectory)
    #     return trace
    def runModel(self, x, goal, dt, time_bound, mode, lane_map):
        t = 0
        time = [t]
        trace = [[t]]
        trace[0].extend(x)
        i = 0
        df = 0
        while t<time_bound and df == 0:
            t = round(t+dt,10)
            curr_position = np.array(x)[[0,4,8]]
            error = goal-curr_position 
            distance = np.sqrt((error**2).sum())
            if distance > 1:
                goal = curr_position + error/distance
            x = odeint(self.cl_nonlinear, x, [0, dt], args=(goal,))[-1].tolist()
            trace.append([t] + x)
            df = self.action_handler(mode, x, lane_map)
        return trace

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap_3d = None) -> np.ndarray:
        # print("TC", initialCondition)
        # total time_bound remained
        time_bound = float(time_bound)
        traces = []
        end_time = 0
        time_limit = lane_map.get_time_limit(self.id)
        time_limit = 2
        mode_parameters = lane_map.get_next_point(
            lane_map.trans_func(mode[1]), self.id, np.array(initialCondition[:3]), np.array(initialCondition[3:6]))
        while time_bound > end_time:
            ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_limit
            ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_limit
            ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_limit
            sym_rot_angle = 0
            # if self.id == 'test1':
            #     print('test', np.linalg.norm(
            #         np.array([ref_vx, ref_vy, ref_vz])))
            # trace = self.runModel(mode, mode_parameters[0:3] + list(initialCondition), min(time_limit, time_bound-end_time), time_step, [ref_vx, ref_vy, ref_vz,
            trace = self.runModel(initialCondition, mode_parameters[3:],time_step, min(time_limit, time_bound-end_time), mode, lane_map)
            # if lane_map.check_guard_box(self.id, np.array(trace[-1][1:])):
            #     mode_parameters = lane_map.get_next_point(
            #         mode[1], self.id, np.array(trace[-1][1:4]), np.array(trace[-1][4:7]))
            # delta = np.array(trace)[:, 1:4]-mode_parameters[3:6]
            # min_dis = float('inf')
            # min_index = 0
            # index = 0
            # for a in delta:
            #     dis = np.linalg.norm(a)
            #     if min_dis > dis:
            #         min_dis = dis
            #         min_index = index
            #     index += 1
            # if self.id == 'test1':
            #     print(index)
            # trace = trace[:index]
            for p in trace:
                p[0] = round(p[0]+end_time, 4)
            end_time = trace[-1][0]
            initialCondition = trace[-1][1:]
            mode_parameters = lane_map.get_next_point(
                lane_map.trans_func(mode[1]), self.id,  None, np.array(initialCondition[3:6]))
            if round(trace[0][0]-0, 4) != 0:
                trace = trace[1:]
            traces.extend(trace)
        # print([ref_vx, ref_vy, ref_vz, sym_rot_angle])
        return np.array(traces)

# import json
# import os
if __name__ == "__main__":
    # path = os.path.abspath(__file__)
    # path=path.replace('tempCodeRunnerFile.py', 'prarm.json')
    # print(path)
    # with open(path, 'r') as f:
    #     prarms = json.load(f)
    # print(prarms)
    agent = QuadrotorAgent('test',code="")
    agent.run_model([0,0,0,0,0,0,0,0,0,0],np.array([5,5,5]),0.1, 10)

