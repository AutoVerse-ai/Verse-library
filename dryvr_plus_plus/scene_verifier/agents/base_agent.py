from dryvr_plus_plus.scene_verifier.code_parser.pythonparser import ControllerAst
from dryvr_plus_plus.scene_verifier.code_parser.parser import ControllerIR, Env


class BaseAgent:
    def __init__(self, id, code = None, file_name = None):  
        # TODO-PARSER: Use ControllerIR instead of ControllerAst
        # self.controller = ControllerAst(code, file_name)
        e = Env.parse(code, file_name)
        self.controller: ControllerIR = e.to_ir()
        self.id = id

    def TC_simulate(self, mode, initialSet, time_horizon, time_step, map=None):
        raise NotImplementedError