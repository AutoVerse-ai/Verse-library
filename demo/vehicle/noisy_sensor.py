from verse.sensor import BaseSensor
from typing import Tuple
import numpy as np

class NoisyVehicleSensor(BaseSensor):
    def __init__(self, noise_x = Tuple[float, float], noise_y = Tuple[float, float]):
        self.noise_x = noise_x
        self.noise_y = noise_y

    def sense(self, scenario, agent, state_dict, lane_map):
        cont, disc, len_dict = super().sense(scenario, agent, state_dict, lane_map)
        tmp = np.array(list(state_dict.values())[0][0])
        if tmp.ndim<2:
            return cont, disc, len_dict
        else:
            # Apply noise to observed x
            if 'others.x' in cont:
                for x_range in cont['others.x']:
                    x_range[0] -= self.noise_x[0]
                    x_range[1] += self.noise_x[1]

            # Apply noise to observed y
            if 'others.y' in cont:
                for y_range in cont['others.y']:
                    y_range[0] -= self.noise_y[0]
                    y_range[1] += self.noise_y[1]
            return cont, disc, len_dict
