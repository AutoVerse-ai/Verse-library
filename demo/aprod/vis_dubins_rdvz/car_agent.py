# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from verse.utils.utils import wrap_to_pi
from verse.analysis.analysis_tree import TraceType
from verse.parser import ControllerIR


class NPCAgent(BaseAgent):
    def __init__(self, id, initial_state=None, initial_mode=None):
        self.id = id
        self.decision_logic = ControllerIR.empty()
        self.set_initial_state(initial_state)
        self.set_initial_mode(initial_mode)
        self.set_static_parameter(None)
        self.set_uncertain_parameter(None)

    @staticmethod
    def dynamics(t, state, u):
        theta, v = state[2:4]
        delta, a = u
        x_dot = v * np.cos(theta + delta)
        y_dot = v * np.sin(theta + delta)
        theta_dot = v / 1.75 * np.sin(delta)
        v_dot = a
        dots = [x_dot, y_dot, theta_dot, v_dot]
        if len(state) == 4:
            return dots
        return dots + [1]

    def action_handler(self, mode, state, lane_map: LaneMap) -> Tuple[float, float]:
        """Computes steering and acceleration based on current lane, target lane and
        current state using a Stanley controller-like rule"""
        x, y, theta, v = state[:4]
        vehicle_mode = mode[0]
        vehicle_lane = mode[1]
        vehicle_pos = np.array([x, y])
        d = -lane_map.get_lateral_distance(vehicle_lane, vehicle_pos)
        psi = lane_map.get_lane_heading(vehicle_lane, vehicle_pos) - theta
        steering = psi + np.arctan2(0.45 * d, v)
        steering = np.clip(steering, -0.61, 0.61)
        a = 0
        return steering, a

    def TC_simulate(
        self, mode: Tuple[str], init, time_bound, time_step, lane_map: LaneMap = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        for i in range(num_points):
            steering, a = self.action_handler(mode, init, lane_map)
            r = ode(self.dynamics)
            r.set_initial_value(init).set_f_params([steering, a])
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten()
            if init[3] < 0:
                init[3] = 0
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace


class CarAgent(BaseAgent):
    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state=None,
        initial_mode=None,
        speed: float = 1,
        accel: float = 1,
    ):
        super().__init__(
            id, code, file_name, initial_state=initial_state, initial_mode=initial_mode
        )
        self.speed = speed
        self.accel = accel

    @staticmethod
    def dynamics(t, state, u):
        x, y, theta, v, hx, hy, htheta, hv = state
        delta, a = u
        x_dot = v * np.cos(theta + delta)
        y_dot = v * np.sin(theta + delta)
        theta_dot = v / 1.75 * np.sin(delta)
        v_dot = a
        hx_dot = hv * np.cos(htheta + delta)
        hy_dot = hv * np.sin(htheta + delta)
        htheta_dot = hv / 1.75 * np.sin(delta)
        hv_dot = a
        return [x_dot, y_dot, theta_dot, v_dot,
                hx_dot, hy_dot, htheta_dot, hv_dot]

    def action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        # x, y, theta, v, hx, hy, htheta, hv = state
        vehicle_mode, _ = mode
        # vehicle_pos = np.array([hx, hy]) # using observed rather than true position
        a = 0

        # deg_to_rad = np.pi/180
        steering = 0
        omega = 2 # let omega be 2 radians/s 
        if vehicle_mode == "Normal":
            pass
        elif vehicle_mode == "Left":
            steering = omega
        elif vehicle_mode == "Right":
            steering = -omega
        else:
            raise Exception(f'Unexpected vehicle mode: {vehicle_mode}')

        return steering, a

    def TC_simulate(
        self, mode: List[str], init, time_bound, time_step, lane_map: LaneMap = None
    ) -> TraceType:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        init = np.array(init)
        timer_start = init[12] # or 5 if estimated and error states unused
        holder = init[13:] # contains id, connected_ids, assigned_id, dist, prev_sense, and cur_sense that shouldn't be touched 
        augmented_init = np.concatenate([init[:4], init[:4]-init[8:12]]) # [x,hx] -- note: never sample hx directly, it will introduce extra errors
        for i in range(num_points):
            steering, a = self.action_handler(mode, augmented_init, lane_map)
            r = ode(self.dynamics)
            r.set_initial_value(augmented_init).set_f_params([steering, a])
            res: np.ndarray = r.integrate(r.t + time_step)
            augmented_init = res.flatten() # this init is [x,hx]
            augmented_init[3] = 0 if augmented_init[3] < 0 else augmented_init[3]
            augmented_init[7] = 0 if augmented_init[7] < 0 else augmented_init[7]
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = np.concatenate([augmented_init, augmented_init[:4]-augmented_init[4:8], [timer_start+time_step * (i + 1)], holder])
        return trace


class WeirdCarAgent(CarAgent):
    def __init__(self, id, code=None, file_name=None):
        super().__init__(id, code, file_name)
        self.gain = 0.2
        self.thres = 0.8

    def action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        vehicle_mode = mode[0]
        steering, a = self._action_handler(mode, state, lane_map)
        # print("agnt", vehicle_mode, state, steering, a)
        return steering, a

    def _action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        vehicle_mode = mode[0]
        theta = state[2]
        if abs(theta) > self.thres:
            return 0, 0
        if vehicle_mode == "SwitchLeft" and theta >= 0:
            return self.gain, 0
        elif vehicle_mode == "SwitchRight" and theta <= 0:
            return -self.gain, 0
        else:
            return 0, 0


class CarAgentDebounced(CarAgent):
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
            id,
            code,
            file_name,
            initial_state=initial_state,
            initial_mode=initial_mode,
            speed=speed,
            accel=accel,
        )

    @staticmethod
    def dynamics(t, state, u):
        return super(CarAgentDebounced, CarAgentDebounced).dynamics(t, state[:4], u) + [1]

    def action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        return super(CarAgentDebounced, self).action_handler(
            mode, state[:4], lane_map
        )


class CarAgentSwitch2(CarAgent):
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
            id,
            code,
            file_name,
            initial_state=initial_state,
            initial_mode=initial_mode,
            speed=speed,
            accel=accel,
        )

    @staticmethod
    def dynamics(t, state, u):
        return super(CarAgentSwitch2, CarAgentSwitch2).dynamics(t, state[:4], u) + [1]

    def action_handler(self, mode: List[str], state, lane_map: LaneMap) -> Tuple[float, float]:
        x, y, theta, v, _ = state
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
        elif vehicle_mode == "SwitchLeft2":
            d += lane_width * 2
        elif vehicle_mode == "SwitchRight2":
            d -= lane_width * 2
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
