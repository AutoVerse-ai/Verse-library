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


class QuadrotorAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None, t_v_pair=[], box_side=[]):
        super().__init__(id, code, file_name)
        self.t_v_pair = t_v_pair
        self.box_side = box_side

    @staticmethod
    def dynamic(t, state, u):
        u1, u2, u3, bx, by, bz, sc, ddf = u  # len 7
        vx, vy, vz = state[6:]  # len 9
        sc = -1 * sc
        dvx = 9.81 * np.sin(u1) / np.cos(u1)
        dvy = -9.81 * np.sin(u2) / np.cos(u2)
        tmp1 = dvx * math.cos(sc) - dvy * math.sin(sc)
        tmp2 = dvx * math.sin(sc) + dvy * math.cos(sc)
        dvx = tmp1
        dvy = tmp2
        dvz = u3 - 9.81
        dx = vx
        dy = vy
        dz = vz
        dref_x = bx
        dref_y = by
        dref_z = bz
        return [dref_x, dref_y, dref_z, dx, dy, dz, dvx, dvy, dvz]

    def action_handler(self, mode, state, lane_map: LaneMap_3d):
        # if mode[0] == 'Normal':
        df = 0
        if lane_map.check_guard_box(self.id, state[3:9], self.box_side):
            # lane_map.get_next_point(mode[1], self.id, state[3:6])
            df = 1
        # else:
        #     raise ValueError
        return df

    def runModel(
        self, mode, initalCondition, time_bound, time_step, ref_input, lane_map: LaneMap_3d
    ):
        path = os.path.abspath(__file__)
        path = path.replace("quadrotor_agent.py", "prarm.json")
        # print(path)
        with open(path, "r") as f:
            prarms = json.load(f)
        bias1 = prarms["bias1"]
        bias2 = prarms["bias2"]
        bias3 = prarms["bias3"]
        weight1 = prarms["weight1"]
        weight2 = prarms["weight2"]
        weight3 = prarms["weight3"]

        bias1 = torch.FloatTensor(bias1)
        bias2 = torch.FloatTensor(bias2)
        bias3 = torch.FloatTensor(bias3)
        weight1 = torch.FloatTensor(weight1)
        weight2 = torch.FloatTensor(weight2)
        weight3 = torch.FloatTensor(weight3)
        controller = FFNNC()
        controller.layer1.weight = torch.nn.Parameter(weight1)
        controller.layer2.weight = torch.nn.Parameter(weight2)
        controller.layer3.weight = torch.nn.Parameter(weight3)
        controller.layer1.bias = torch.nn.Parameter(bias1)
        controller.layer2.bias = torch.nn.Parameter(bias2)
        controller.layer3.bias = torch.nn.Parameter(bias3)
        control_input_list = [
            [-0.1, -0.1, 7.81],
            [-0.1, -0.1, 11.81],
            [-0.1, 0.1, 7.81],
            [-0.1, 0.1, 11.81],
            [0.1, -0.1, 7.81],
            [0.1, -0.1, 11.81],
            [0.1, 0.1, 7.81],
            [0.1, 0.1, 11.81],
        ]
        init = initalCondition
        trajectory = [init]
        r = ode(self.dynamic)
        # r.set_initial_value(init)
        ex_list = []
        ey_list = []
        ez_list = []
        t = 0
        time = [t]
        trace = [[t]]
        trace[0].extend(init[3:])
        i = 0
        df = 0
        while (t < time_bound) or df != 1:
            ex = trajectory[i][3] - trajectory[i][0]
            ey = trajectory[i][4] - trajectory[i][1]
            ez = trajectory[i][5] - trajectory[i][2]
            # if self.id == 'test1':
            #     print([t, ex, ey, ez, trajectory[i][0],
            #           trajectory[i][1], trajectory[i][2]])
            evx = trajectory[i][6] - ref_input[0]
            evy = trajectory[i][7] - ref_input[1]
            evz = trajectory[i][8] - ref_input[2]

            sc = ref_input[3]  # math.atan2(dot, det)

            tmp1 = ex * math.cos(sc) - ey * math.sin(sc)
            tmp2 = ex * math.sin(sc) + ey * math.cos(sc)
            ex = tmp1
            ey = tmp2

            tmp1 = evx * math.cos(sc) - evy * math.sin(sc)
            tmp2 = evx * math.sin(sc) + evy * math.cos(sc)
            evx = tmp1
            evy = tmp2

            data = torch.FloatTensor(
                [0.2 * ex, 0.2 * ey, 0.2 * ez, 0.1 * evx, 0.1 * evy, 0.1 * evz]
            )
            res = controller(data)
            res = res.detach().numpy()
            idx = np.argmax(res)
            u = control_input_list[idx] + ref_input[0:3] + [sc]

            df = self.action_handler(mode, init, lane_map)

            u = u + [df]
            init = trajectory[i]  # len 9
            r = ode(self.dynamic)
            r.set_initial_value(init)
            r.set_f_params(u)
            val = r.integrate(r.t + time_step)

            t = t + time_step
            if round(t - time_bound - time_step, 4) >= 0:
                break
            i += 1
            #  print(i,idx,u,res)
            trajectory.append(val)
            time.append(t)

            ex_list.append(ex)
            ey_list.append(ey)
            ez_list.append(ez)
            trace.append([t])
            # remove the reference trajectory from the trace
            trace[i].extend(val[3:])
            # if self.id == 'test1':
            #     print([t, trajectory[i][0], trajectory[i][1], trajectory[i][2]])
            #     print('curr', val[3:6])
        # print(trajectory)
        return trace

    def TC_simulate(
        self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap_3d = None
    ) -> np.ndarray:
        # print("TC", initialCondition)
        # total time_bound remained
        time_bound = float(time_bound)
        traces = []
        end_time = 0
        time_limit = self.t_v_pair[0]
        mode_parameters = lane_map.get_next_point(
            lane_map.trans_func(mode[1]),
            self.id,
            np.array(initialCondition[:3]),
            np.array(initialCondition[3:6]),
            self.t_v_pair,
        )
        while time_bound > end_time:
            ref_vx = (mode_parameters[3] - mode_parameters[0]) / time_limit
            ref_vy = (mode_parameters[4] - mode_parameters[1]) / time_limit
            ref_vz = (mode_parameters[5] - mode_parameters[2]) / time_limit
            sym_rot_angle = 0
            # if self.id == 'test1':
            #     print('test', np.linalg.norm(
            #         np.array([ref_vx, ref_vy, ref_vz])))
            trace = self.runModel(
                mode,
                mode_parameters[0:3] + list(initialCondition),
                min(time_limit, time_bound - end_time),
                time_step,
                [ref_vx, ref_vy, ref_vz, sym_rot_angle],
                lane_map,
            )
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
                p[0] = round(p[0] + end_time, 4)
            end_time = trace[-1][0]
            initialCondition = trace[-1][1:]
            mode_parameters = lane_map.get_next_point(
                lane_map.trans_func(mode[1]),
                self.id,
                None,
                np.array(initialCondition[3:6]),
                self.t_v_pair,
            )
            if round(trace[0][0] - 0, 4) != 0:
                trace = trace[1:]
            traces.extend(trace)
        # print([ref_vx, ref_vy, ref_vz, sym_rot_angle])
        return np.array(traces)


# import json
# import os
# if __name__ == "__main__":
#     path = os.path.abspath(__file__)
#     path=path.replace('tempCodeRunnerFile.py', 'prarm.json')
#     print(path)
#     with open(path, 'r') as f:
#         prarms = json.load(f)
#     print(prarms)
