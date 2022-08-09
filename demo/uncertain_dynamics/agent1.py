from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse.agents import BaseAgent
from verse.map import LaneMap
from verse.parser import ControllerIR

class Agent1(BaseAgent):
    def __init__(self, id):
        self.id = id 
        self.controller = ControllerIR.empty()