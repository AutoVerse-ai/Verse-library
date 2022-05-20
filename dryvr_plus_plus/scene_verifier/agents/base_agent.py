from dryvr_plus_plus.scene_verifier.code_parser.pythonparser import ControllerAst

class BaseAgent:
    def __init__(self, id, code = None, file_name = None):  
        self.controller = ControllerAst(code, file_name)
        self.id = id

    def TC_simulate(self, mode, initialSet, time_horizon, map=None):
        raise NotImplementedError