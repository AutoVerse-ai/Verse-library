# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
import torch

from tutorial_utils import drone_params
from verse import BaseAgent
from verse import LaneMap
from verse.map.lane_map_3d import LaneMap_3d
from verse.analysis.utils import wrap_to_pi
from verse.analysis.analysis_tree import TraceType


class CarAgent(BaseAgent):
    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state=None,
        initial_mode=None,
        speed: float = 2,
        accel: float = 1,
    ):
        super().__init__(
            id, code, file_name, initial_state=initial_state, initial_mode=initial_mode
        )
        self.speed = speed
        self.accel = accel

    @staticmethod
    def dynamic(t, state, u):
        x, y, theta, v = state
        delta, a = u
        x_dot = v * np.cos(theta + delta)
        y_dot = v * np.sin(theta + delta)
        theta_dot = v / 1.75 * np.sin(delta)
        v_dot = a
        return [x_dot, y_dot, theta_dot, v_dot]

    def action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        x, y, theta, v = state
        vehicle_mode, vehicle_lane = mode
        vehicle_pos = np.array([x, y])
        a = 0
        lane_width = lane_map.get_lane_width(vehicle_lane)
        d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
        if vehicle_mode == "Normal" or vehicle_mode == "Stop":
            pass
        elif vehicle_mode == "SwitchLeft":
            d += lane_width
        elif vehicle_mode == "SwitchRight":
            d -= lane_width
        elif vehicle_mode == "Brake":
            a = max(-self.accel, -v)
        elif vehicle_mode == "Accel":
            a = min(self.accel, self.speed - v)
        else:
            raise ValueError(f"Invalid mode: {vehicle_mode}")

        heading = lane_map.get_lane_heading(vehicle_lane, vehicle_pos)
        psi = wrap_to_pi(heading - theta)
        steering = psi + np.arctan2(0.45 * d, v)
        steering = np.clip(steering, -0.61, 0.61)
        return steering, a

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map: LaneMap = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            steering, a = self.action_handler(mode, init, lane_map)
            r = ode(self.dynamic)
            r.set_initial_value(init).set_f_params([steering, a])
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            if init[3] < 0:
                init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

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


class DroneAgent(BaseAgent):
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
        tmp1 = dvx * np.cos(sc) - dvy * np.sin(sc)
        tmp2 = dvx * np.sin(sc) + dvy * np.cos(sc)
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

    def action_handler(self, mode, state, track_map: LaneMap_3d):
        # if mode[0] == 'Normal':
        df = 0
        if track_map.check_guard_box(self.id, state[3:9], self.box_side):
            # track_map.get_next_point(mode[1], self.id, state[3:6])
            df = 1
        # else:
        #     raise ValueError
        return df

    def runModel(
        self, mode, initalCondition, time_bound, time_step, ref_input, track_map: LaneMap_3d
    ):
        # path = os.path.abspath(__file__)
        # path = path.replace('quadrotor_agent.py', 'prarm.json')
        # # print(path)
        # with open(path, 'r') as f:
        #     prarms = json.load(f)
        params = drone_params
        bias1 = params["bias1"]
        bias2 = params["bias2"]
        bias3 = params["bias3"]
        weight1 = params["weight1"]
        weight2 = params["weight2"]
        weight3 = params["weight3"]

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

            sc = ref_input[3]  # np.atan2(dot, det)

            tmp1 = ex * np.cos(sc) - ey * np.sin(sc)
            tmp2 = ex * np.sin(sc) + ey * np.cos(sc)
            ex = tmp1
            ey = tmp2

            tmp1 = evx * np.cos(sc) - evy * np.sin(sc)
            tmp2 = evx * np.sin(sc) + evy * np.cos(sc)
            evx = tmp1
            evy = tmp2

            data = torch.FloatTensor(
                [0.2 * ex, 0.2 * ey, 0.2 * ez, 0.1 * evx, 0.1 * evy, 0.1 * evz]
            )
            res = controller(data)
            res = res.detach().numpy()
            idx = np.argmax(res)
            u = control_input_list[idx] + ref_input[0:3] + [sc]

            df = self.action_handler(mode, init, track_map)

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
        return trace

    def TC_simulate(
        self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap_3d = None
    ) -> np.ndarray:
        time_bound = float(time_bound)
        traces = []
        end_time = 0
        time_limit = self.t_v_pair[0]
        mode_parameters = track_map.get_next_point(
            track_map.trans_func(mode[1]),
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
            trace = self.runModel(
                mode,
                mode_parameters[0:3] + list(initialCondition),
                min(time_limit, time_bound - end_time),
                time_step,
                [ref_vx, ref_vy, ref_vz, sym_rot_angle],
                track_map,
            )
            for p in trace:
                p[0] = round(p[0] + end_time, 4)
            end_time = trace[-1][0]
            initialCondition = trace[-1][1:]
            mode_parameters = track_map.get_next_point(
                track_map.trans_func(mode[1]),
                self.id,
                None,
                np.array(initialCondition[3:6]),
                self.t_v_pair,
            )
            if round(trace[0][0] - 0, 4) != 0:
                trace = trace[1:]
            traces.extend(trace)
        return np.array(traces)
