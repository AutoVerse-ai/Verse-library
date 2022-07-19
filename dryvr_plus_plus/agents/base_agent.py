from dryvr_plus_plus.code_parser import ControllerIR


class BaseAgent:
    def __init__(self, id, code = None, file_name = None):  
        self.controller: ControllerIR = ControllerIR.parse(code, file_name)
        self.id = id

    def TC_simulate(self, mode, initialSet, time_horizon, time_step, map=None):
        raise NotImplementedError
