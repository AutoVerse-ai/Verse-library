from typing import DefaultDict, Optional, Tuple, List, Dict, Any
import copy
from dataclasses import dataclass
import numpy as np
from pprint import pp

from verse.agents.base_agent import BaseAgent
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree
from verse.analysis.utils import dedup, sample_rect
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

EGO, OTHERS = "ego", "others"

def check_ray_init(parallel: bool) -> None:
    if parallel:
        import ray
        if not ray.is_initialized():
            ray.init()

def red(s):
    return "\x1b[31m" + s + "\x1b[0m"

@dataclass(frozen=True)
class ScenarioConfig:
    incremental: bool = False
    unsafe_continue: bool = False
    init_seg_length: int = 1000
    reachability_method: str = 'DRYVR'
    parallel_sim_ahead: int = 8
    parallel_ver_ahead: int = 8
    parallel: bool = True

class Scenario:
    def __init__(self, config=ScenarioConfig()):
        self.agent_dict: Dict[str, BaseAgent] = {}
        self.simulator = Simulator(config)
        self.verifier = Verifier(config)
        self.init_dict = {}
        self.init_mode_dict = {}
        self.static_dict = {}
        self.uncertain_param_dict = {}
        self.map = LaneMap()
        self.sensor = BaseSensor()
        self.past_runs = []

        # Parameters
        self.config = config

    def update_config(self, config):
        self.config = config
        self.verifier.config = config
        self.simulator.config = config

    def set_sensor(self, sensor):
        self.sensor = sensor

    def set_map(self, track_map: LaneMap):
        self.map = track_map
        # Update the lane mode field in the agent
        for agent_id in self.agent_dict:
            agent = self.agent_dict[agent_id]
            self.update_agent_lane_mode(agent, track_map)

    def add_agent(self, agent: BaseAgent):
        if self.map is not None:
            # Update the lane mode field in the agent
            self.update_agent_lane_mode(agent, self.map)
        self.agent_dict[agent.id] = agent
        if hasattr(agent, 'init_cont') and agent.init_cont is not None:
            self.init_dict[agent.id] = copy.deepcopy(agent.init_cont) 
        if hasattr(agent, 'init_disc') and agent.init_disc is not None:
            self.init_mode_dict[agent.id] = copy.deepcopy(agent.init_disc)

        if hasattr(agent, 'static_parameters') and agent.static_parameters is not None:
            self.static_dict[agent.id] = copy.deepcopy(agent.static_parameters)
        else:
            self.static_dict[agent.id] = []
        if hasattr(agent, 'uncertain_parameters') and agent.uncertain_parameters is not None:
            self.uncertain_param_dict[agent.id] = copy.deepcopy(agent.uncertain_parameters)
        else:
            self.uncertain_param_dict[agent.id] = []


    # TODO-PARSER: update this function
    def update_agent_lane_mode(self, agent: BaseAgent, track_map: LaneMap):
        for lane_id in track_map.lane_dict:
            if 'TrackMode' in agent.decision_logic.mode_defs and lane_id not in agent.decision_logic.mode_defs['TrackMode'].modes:
                agent.decision_logic.mode_defs['TrackMode'].modes.append(lane_id)
        # mode_vals = list(agent.decision_logic.modes.values())
        # agent.decision_logic.vertices = list(itertools.product(*mode_vals))
        # agent.decision_logic.vertexStrings = [','.join(elem) for elem in agent.decision_logic.vertices]

    def set_init_single(self, agent_id, init: list, init_mode: tuple, static=[], uncertain_param=[]):
        assert agent_id in self.agent_dict, 'agent_id not found'
        agent = self.agent_dict[agent_id]
        assert len(init) == 1 or len(
            init) == 2, 'the length of init should be 1 or 2'
        # print(agent.decision_logic.state_defs.values())
        if agent.decision_logic != agent.decision_logic.empty():
            for i in init:
                assert len(i) == len(
                    list(agent.decision_logic.state_defs.values())[0].cont),  'the length of element in init not fit the number of continuous variables'
            # print(agent.decision_logic.mode_defs)
            assert len(init_mode) == len(
                list(agent.decision_logic.state_defs.values())[0].disc),  'the length of element in init_mode not fit the number of discrete variables'
        if len(init) == 1:
            init = init+init
        self.init_dict[agent_id] = copy.deepcopy(init)
        self.init_mode_dict[agent_id] = copy.deepcopy(init_mode)
        self.agent_dict[agent_id].set_initial(init, init_mode)
        if static:
            self.static_dict[agent_id] = copy.deepcopy(static)
            self.agent_dict[agent_id].set_static_parameter(static)
        else:
            self.static_dict[agent_id] = []
        if uncertain_param:
            self.uncertain_param_dict[agent_id] = copy.deepcopy(
                uncertain_param)
            self.agent_dict[agent_id].set_uncertain_parameter(uncertain_param)
        else:
            self.uncertain_param_dict[agent_id] = []
        return

    def set_init(self, init_list, init_mode_list, static_list=[], uncertain_param_list=[]):
        assert len(init_list) == len(
            self.agent_dict), 'the length of init_list not fit the number of agents'
        assert len(init_mode_list) == len(
            self.agent_dict), 'the length of init_mode_list not fit the number of agents'
        assert len(static_list) == len(
            self.agent_dict) or len(static_list) == 0, 'the length of static_list not fit the number of agents or equal to 0'
        assert len(uncertain_param_list) == len(self.agent_dict)\
            or len(uncertain_param_list) == 0, 'the length of uncertain_param_list not fit the number of agents or equal to 0'
        print(init_mode_list)
        print(type(init_mode_list))
        if not static_list:
            static_list = [[] for i in range(0, len(self.agent_dict))]
            # print(static_list)
        if not uncertain_param_list:
            uncertain_param_list = [[] for i in range(0, len(self.agent_dict))]
            # print(uncertain_param_list)
        for i, agent_id in enumerate(self.agent_dict.keys()):
            self.set_init_single(agent_id, init_list[i],
                                 init_mode_list[i], static_list[i], uncertain_param_list[i])

    def check_init(self):
        for agent_id in self.agent_dict.keys():
            assert agent_id in self.init_dict, 'init of {} not initialized'.format(
                agent_id)
            assert agent_id in self.init_mode_dict, 'init_mode of {} not initialized'.format(
                agent_id)
            assert agent_id in self.static_dict, 'static of {} not initialized'.format(
                agent_id)
            assert agent_id in self.uncertain_param_dict, 'uncertain_param of {} not initialized'.format(
                agent_id)
        return

    def simulate_multi(self, time_horizon, num_sim):
        res_list = []
        for i in range(num_sim):
            trace = self.simulate(time_horizon)
            res_list.append(trace)
        return res_list

    def simulate(self, time_horizon, time_step, max_height=None, seed=None) -> AnalysisTree:
        check_ray_init(self.config.parallel)
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        # agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init_list.append(sample_rect(self.init_dict[agent_id], seed))
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            # agent_list.append(self.agent_dict[agent_id])
        print(init_list)
        tree = self.simulator.simulate(init_list, init_mode_list, static_list, uncertain_param_list, self.agent_dict, self.sensor, time_horizon, time_step, max_height, self.map, len(self.past_runs), self.past_runs)
        self.past_runs.append(tree)
        return tree

    def simulate_simple(self, time_horizon, time_step, max_height=None, seed = None) -> AnalysisTree:
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init_list.append(sample_rect(self.init_dict[agent_id], seed))
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        print(init_list)
        tree = self.simulator.simulate_simple(init_list, init_mode_list, static_list, uncertain_param_list, agent_list, time_horizon, time_step, max_height, self.map, self.sensor, len(self.past_runs), self.past_runs)
        self.past_runs.append(tree)
        return tree

    def verify(self, time_horizon, time_step, max_height=None, params={}) -> AnalysisTree:
        check_ray_init(self.config.parallel)
        self.check_init()
        init_list = []
        init_mode_list = []
        static_list = []
        agent_list = []
        uncertain_param_list = []
        for agent_id in self.agent_dict:
            init = self.init_dict[agent_id]
            tmp = np.array(init)
            if tmp.ndim < 2:
                init = [init, init]
            init_list.append(init)
            init_mode_list.append(self.init_mode_dict[agent_id])
            static_list.append(self.static_dict[agent_id])
            uncertain_param_list.append(self.uncertain_param_dict[agent_id])
            agent_list.append(self.agent_dict[agent_id])
        
        tree = self.verifier.compute_full_reachtube(init_list, init_mode_list, static_list, uncertain_param_list, self.agent_dict, self.sensor, time_horizon,
                                                    time_step, max_height, self.map, self.config.init_seg_length, self.config.reachability_method, len(self.past_runs), self.past_runs, params)
        self.past_runs.append(tree)
        return tree

@dataclass
class ExprConfig:
    config: ScenarioConfig
    args: str
    rest: List[str]
    compare: bool = False
    plot: bool = False
    dump: bool = False
    sim: bool = True

    @staticmethod
    def from_arg(a: List[str], **kw) -> "ExprConfig":
        arg = "" if len(a) < 2 else a[1]
        sconfig = ScenarioConfig(incremental='i' in arg, parallel='l' in arg, **kw)
        cpds = "c" in arg, "p" in arg, "d" in arg, "v" not in arg
        for o in "cilpdv":
            arg = arg.replace(o, "")
        expconfig = ExprConfig(sconfig, arg, a[2:], *cpds)
        expconfig.kw=kw
        return expconfig
    def disp(self):
        print('args', self.args)
        print('rest', self.rest)
        print('compare', self.compare)
        print('plot', self.plot)
        print('dump', self.dump)
        print('sim', self.sim)

from pympler import asizeof
import timeit

class Benchmark:
    agent_type: str
    num_agent: int
    map_name: str
    cont_engine: str
    noisy_s: str
    num_nodes: int
    run_time: float
    cache_size: float
    cache_hits: Tuple[int, int]
    _start_time: float

    def __init__(self, argv: List[str], **kw):
        self.config = ExprConfig.from_arg(argv, **kw)
        # self.config.disp()
        self.scenario = Scenario(self.config.config)

    def run(self, *a, **kw)->AnalysisTree:
        f = self.scenario.simulate if self.config.sim else self.scenario.verify
        self.cont_engine = self.scenario.config.reachability_method
        self._start_time = timeit.default_timer()
        self.traces = f(*a, **kw)
        self.run_time = timeit.default_timer() - self._start_time
        if self.config.sim:
            self.cache_size = asizeof.asizeof(self.scenario.simulator.cache) / 1_000_000
            self.cache_hits = self.scenario.simulator.cache_hits
        else:
            self.cache_size = (asizeof.asizeof(self.scenario.verifier.cache) + asizeof.asizeof(self.scenario.verifier.trans_cache)) / 1_000_000
            self.cache_hits = (self.scenario.verifier.tube_cache_hits[0] + self.scenario.verifier.trans_cache_hits[0], self.scenario.verifier.tube_cache_hits[1] + self.scenario.verifier.trans_cache_hits[1])
        self.num_agent = len(self.scenario.agent_dict)
        self.map_name = self.scenario.map.__class__.__name__
        if self.map_name == 'LaneMap':
            self.map_name = 'N/A'
        self.num_nodes = len(self.traces.nodes)
        return self.traces

    def compare_run(self, *a, **kw):
        assert self.config.compare
        traces1 = self.run( *a, **kw)
        self.report()
        if len(self.config.rest) == 0:
            traces2 = self.run( *a, **kw)
        else:
            arg = self.config.rest[0]
            self.config.config = ScenarioConfig(incremental='i' in arg, parallel='l' in arg, **self.config.kw)
            self.scenario.update_config(self.config.config)
            traces2 = self.run( *a, **kw)
        self.report()
        print("trace1 contains trace2?", traces1.contains(traces2))
        print("trace2 contains trace1?", traces2.contains(traces1))
        return traces1, traces2

    def report(self):
        print_dict={"#A": self.num_agent,
                    "A": self.agent_type,
                    "Map": self.map_name,
                    "postCont": self.cont_engine,
                    "Noisy S": self.noisy_s,
                    "# Tr": self.num_nodes,
                    "Run Time": self.run_time
                    }
        print(print_dict)
        print("report:")
        print("#agents:", self.num_agent)
        print("map name:", self.map_name)
        print("#nodes:", self.num_nodes)
        print(f"run time: {self.run_time:.2f}s")
        if self.config.config.incremental:
            print(f"cache size: {self.cache_size:.2f}MB")
            print(f"cache hit rate: {(self.cache_hits[0], self.cache_hits[1])}")
            print(f"cache hit rate: {self.cache_hits[0] / (self.cache_hits[0] + self.cache_hits[1]) * 100:.2f}%")
