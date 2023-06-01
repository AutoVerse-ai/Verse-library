from typing import Tuple, List, Dict
import copy
from dataclasses import dataclass
import numpy as np

from verse.agents.base_agent import BaseAgent
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree, ReachabilityMethod
from verse.analysis.analysis_tree import AnalysisTreeNodeType
from verse.analysis.utils import sample_rect
from verse.parser.parser import ControllerIR
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

EGO, OTHERS = "ego", "others"


def _check_ray_init(parallel: bool) -> None:
    if parallel:
        import ray

        if not ray.is_initialized():
            ray.init()


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for how simulation/verification is performed for a scenario. Properties are
    immutable so that incremental verification works correctly."""

    incremental: bool = False
    """Enable incremental simulation/verification. Results from previous runs will be used to try to
    speed up experiments. Result is undefined when the map, agent dynamics and sensor are changed."""
    unsafe_continue: bool = False
    """Continue exploring the branch when an unsafe condition occurs."""
    init_seg_length: int = 1000
    reachability_method: ReachabilityMethod = ReachabilityMethod.DRYVR
    """Method of performing reachability. Can be DryVR, NeuReach, MixMonoCont and MixMonoDisc."""
    parallel_sim_ahead: int = 8
    """The number of simulation tasks to dispatch before waiting."""
    parallel_ver_ahead: int = 8
    """The number of verification tasks to dispatch before waiting."""
    parallel: bool = True
    """Enable parallelization. Uses the Ray library. Could be slower for small scenarios."""
    try_local: bool = False
    """Heuristic. When enabled, try to use the local thread when some results are cached."""


class Scenario:
    """A simulation/verification scenario."""

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

    def cleanup_cache(self):
        self.past_runs = []
        self.simulator = Simulator(self.config)
        self.verifier = Verifier(self.config)

    def update_config(self, config):
        self.config = config
        self.verifier.config = config
        self.simulator.config = config

    def set_sensor(self, sensor):
        """Sets the sensor for the scenario. Will use the default sensor when not called."""
        self.sensor = sensor

    def set_map(self, track_map: LaneMap):
        """Sets the map for the scenario."""
        self.map = track_map
        # Update the lane mode field in the agent
        for agent_id in self.agent_dict:
            agent = self.agent_dict[agent_id]
            self._update_agent_lane_mode(agent, track_map)

    def add_agent(self, agent: BaseAgent):
        """Adds an agent to the scenario."""
        if self.map is not None:
            # Update the lane mode field in the agent
            self._update_agent_lane_mode(agent, self.map)
        self.agent_dict[agent.id] = agent
        if hasattr(agent, "init_cont") and agent.init_cont is not None:
            self.init_dict[agent.id] = copy.deepcopy(agent.init_cont)
        if hasattr(agent, "init_disc") and agent.init_disc is not None:
            self.init_mode_dict[agent.id] = copy.deepcopy(agent.init_disc)

        if hasattr(agent, "static_parameters") and agent.static_parameters is not None:
            self.static_dict[agent.id] = copy.deepcopy(agent.static_parameters)
        else:
            self.static_dict[agent.id] = []
        if hasattr(agent, "uncertain_parameters") and agent.uncertain_parameters is not None:
            self.uncertain_param_dict[agent.id] = copy.deepcopy(agent.uncertain_parameters)
        else:
            self.uncertain_param_dict[agent.id] = []

    # TODO-PARSER: update this function
    def _update_agent_lane_mode(self, agent: BaseAgent, track_map: LaneMap):
        for lane_id in track_map.lane_dict:
            if (
                "TrackMode" in agent.decision_logic.mode_defs
                and lane_id not in agent.decision_logic.mode_defs["TrackMode"].modes
            ):
                agent.decision_logic.mode_defs["TrackMode"].modes.append(lane_id)
        # mode_vals = list(agent.decision_logic.modes.values())
        # agent.decision_logic.vertices = list(itertools.product(*mode_vals))
        # agent.decision_logic.vertexStrings = [','.join(elem) for elem in agent.decision_logic.vertices]

    def set_init_single(
        self, agent_id, init: list, init_mode: tuple, static=[], uncertain_param=[]
    ):
        """Sets the initial conditions for a single agent."""
        assert agent_id in self.agent_dict, "agent_id not found"
        agent = self.agent_dict[agent_id]
        assert len(init) == 1 or len(init) == 2, "the length of init should be 1 or 2"
        # print(agent.decision_logic.state_defs.values())
        if agent.decision_logic != agent.decision_logic.empty():
            for i in init:
                assert len(i) == len(
                    list(agent.decision_logic.state_defs.values())[0].cont
                ), "the length of element in init not fit the number of continuous variables"
            # print(agent.decision_logic.mode_defs)
            assert len(init_mode) == len(
                list(agent.decision_logic.state_defs.values())[0].disc
            ), "the length of element in init_mode not fit the number of discrete variables"
        if len(init) == 1:
            init = init + init
        self.init_dict[agent_id] = copy.deepcopy(init)
        self.init_mode_dict[agent_id] = copy.deepcopy(init_mode)
        self.agent_dict[agent_id].set_initial(init, init_mode)
        if static:
            self.static_dict[agent_id] = copy.deepcopy(static)
            self.agent_dict[agent_id].set_static_parameter(static)
        else:
            self.static_dict[agent_id] = []
        if uncertain_param:
            self.uncertain_param_dict[agent_id] = copy.deepcopy(uncertain_param)
            self.agent_dict[agent_id].set_uncertain_parameter(uncertain_param)
        else:
            self.uncertain_param_dict[agent_id] = []
        return

    def set_init(self, init_list, init_mode_list, static_list=[], uncertain_param_list=[]):
        """Sets the initial conditions for all agents. The order will be the same as the order in
        which the agents are added."""
        assert len(init_list) == len(
            self.agent_dict
        ), "the length of init_list not fit the number of agents"
        assert len(init_mode_list) == len(
            self.agent_dict
        ), "the length of init_mode_list not fit the number of agents"
        assert (
            len(static_list) == len(self.agent_dict) or len(static_list) == 0
        ), "the length of static_list not fit the number of agents or equal to 0"
        assert (
            len(uncertain_param_list) == len(self.agent_dict) or len(uncertain_param_list) == 0
        ), "the length of uncertain_param_list not fit the number of agents or equal to 0"
        print(init_mode_list)
        print(type(init_mode_list))
        if not static_list:
            static_list = [[] for i in range(0, len(self.agent_dict))]
            # print(static_list)
        if not uncertain_param_list:
            uncertain_param_list = [[] for i in range(0, len(self.agent_dict))]
            # print(uncertain_param_list)
        for i, agent_id in enumerate(self.agent_dict.keys()):
            self.set_init_single(
                agent_id, init_list[i], init_mode_list[i], static_list[i], uncertain_param_list[i]
            )

    def _check_init(self):
        for agent_id in self.agent_dict.keys():
            assert agent_id in self.init_dict, "init of {} not initialized".format(agent_id)
            assert agent_id in self.init_mode_dict, "init_mode of {} not initialized".format(
                agent_id
            )
            assert agent_id in self.static_dict, "static of {} not initialized".format(agent_id)
            assert (
                agent_id in self.uncertain_param_dict
            ), "uncertain_param of {} not initialized".format(agent_id)
        return

    def simulate_multi(self, time_horizon, num_sim):
        res_list = []
        for i in range(num_sim):
            trace = self.simulate(time_horizon)
            res_list.append(trace)
        return res_list

    def simulate(self, time_horizon, time_step, max_height=None, seed=None) -> AnalysisTree:
        """Compute a single simulation trajectory of the scenario, starting from a single initial state.
        `seed`: the random seed for sampling a point in the region specified by the initial
        conditions"""
        _check_ray_init(self.config.parallel)
        self._check_init()
        root = AnalysisTreeNode.root_from_inits(
            init={aid: sample_rect(init, seed) for aid, init in self.init_dict.items()},
            mode={
                aid: tuple(elem if isinstance(elem, str) else elem.name for elem in modes)
                for aid, modes in self.init_mode_dict.items()
            },
            static={aid: [elem.name for elem in modes] for aid, modes in self.static_dict.items()},
            uncertain_param=self.uncertain_param_dict,
            agent=self.agent_dict,
            type=AnalysisTreeNodeType.SIM_TRACE,
            ndigits=10,
        )
        tree = self.simulator.simulate(
            root,
            self.sensor,
            time_horizon,
            time_step,
            max_height,
            self.map,
            len(self.past_runs),
            self.past_runs,
        )
        self.past_runs.append(tree)
        return tree

    def simulate_simple(self, time_horizon, time_step, max_height=None, seed=None) -> AnalysisTree:
        """Compute the set of reachable states, starting from a single point. Evaluates the decision
        logic code directly, and does not use the internal Python parser and generate
        nondeterministic transitions.
        `seed`: the random seed for sampling a point in the region specified by the initial
        conditions"""
        self._check_init()
        root = AnalysisTreeNode.root_from_inits(
            init={aid: sample_rect(init, seed) for aid, init in self.init_dict.items()},
            mode={
                aid: tuple(elem if isinstance(elem, str) else elem.name for elem in modes)
                for aid, modes in self.init_mode_dict.items()
            },
            static={aid: [elem.name for elem in modes] for aid, modes in self.static_dict.items()},
            uncertain_param=self.uncertain_param_dict,
            agent=self.agent_dict,
            type=AnalysisTreeNodeType.SIM_TRACE,
            ndigits=10,
        )
        tree = self.simulator.simulate_simple(
            root,
            time_horizon,
            time_step,
            max_height,
            self.map,
            self.sensor,
            len(self.past_runs),
            self.past_runs,
        )
        self.past_runs.append(tree)
        return tree

    def verify(self, time_horizon, time_step, max_height=None, params={}) -> AnalysisTree:
        """Compute the set of reachable states, starting from a set of initial states states."""
        _check_ray_init(self.config.parallel)
        self._check_init()
        root = AnalysisTreeNode.root_from_inits(
            init={
                aid: [[init, init] if np.array(init).ndim < 2 else init]
                for aid, init in self.init_dict.items()
            },
            mode={
                aid: tuple(elem if isinstance(elem, str) else elem.name for elem in modes)
                for aid, modes in self.init_mode_dict.items()
            },
            static={aid: [elem.name for elem in modes] for aid, modes in self.static_dict.items()},
            uncertain_param=self.uncertain_param_dict,
            agent=self.agent_dict,
            type=AnalysisTreeNodeType.REACH_TUBE,
            ndigits=10,
        )

        tree = self.verifier.compute_full_reachtube(
            root,
            self.sensor,
            time_horizon,
            time_step,
            max_height,
            self.map,
            self.config.init_seg_length,
            self.config.reachability_method,
            len(self.past_runs),
            self.past_runs,
            params,
        )
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
        sconfig = ScenarioConfig(
            incremental="i" in arg, parallel="l" in arg, try_local="t" in arg, **kw
        )
        cpds = "c" in arg, "p" in arg, "d" in arg, "v" not in arg
        for o in "cilpdv":
            arg = arg.replace(o, "")
        expconfig = ExprConfig(sconfig, arg, a[2:], *cpds)
        expconfig.kw = kw
        return expconfig

    def disp(self):
        print("args", self.args)
        print("rest", self.rest)
        print("compare", self.compare)
        print("plot", self.plot)
        print("dump", self.dump)
        print("sim", self.sim)


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
    leaves: int
    _start_time: float
    parallelness: float
    parallel_time_offset: float
    timesteps: int

    def __init__(self, argv: List[str], **kw):
        self.config = ExprConfig.from_arg(argv, **kw)
        # self.config.disp()
        self.scenario = Scenario(self.config.config)
        self.agent_type = "N/A"
        self.noisy_s = "no"
        self.parallelness = 0
        self.timesteps = 0
        self.parallel_time_offset = 0

    def run(self, *a, **kw) -> AnalysisTree:
        f = self.scenario.simulate if self.config.sim else self.scenario.verify
        self.cont_engine = str(self.scenario.config.reachability_method)
        if len(a) != 2:
            print(f"\x1b[1;31mWARNING: timesteps field may not work ({a})")
        self.timesteps = int(a[0] / a[1])
        self._start_time = timeit.default_timer()
        if self.config.sim:
            self.traces = f(*a)
        else:
            self.traces = f(*a, **kw)
        self.run_time = timeit.default_timer() - self._start_time
        if self.config.sim:
            self.cache_size = asizeof.asizeof(self.scenario.simulator.cache) / 1_000_000
            self.cache_hits = self.scenario.simulator.cache_hits
        else:
            self.cache_size = (
                asizeof.asizeof(self.scenario.verifier.cache)
                + asizeof.asizeof(self.scenario.verifier.trans_cache)
            ) / 1_000_000
            self.cache_hits = (
                self.scenario.verifier.tube_cache_hits[0]
                + self.scenario.verifier.trans_cache_hits[0],
                self.scenario.verifier.tube_cache_hits[1]
                + self.scenario.verifier.trans_cache_hits[1],
            )
        self.num_agent = len(self.scenario.agent_dict)
        self.map_name = self.scenario.map.__class__.__name__
        if self.map_name == "LaneMap":
            self.map_name = "N/A"
        self.num_nodes = len(self.traces.nodes)
        self.leaves = self.traces.leaves()
        if self.config.config.parallel:
            import ray

            parallel_time = (
                sum(ev["dur"] for ev in ray.timeline() if ev["cname"] == "generic_work") / 1_000_000
            )
            self.parallelness = (parallel_time - self.parallel_time_offset) / self.run_time
            self.parallel_time_offset = parallel_time
        return self.traces

    def compare_run(self, *a, **kw):
        assert self.config.compare
        traces1 = self.run(*a, **kw)
        self.report()
        if len(self.config.rest) == 0:
            traces2 = self.run(*a, **kw)
        else:
            # arg = self.config.rest[0]
            # self.config.config = ScenarioConfig(incremental='i' in arg, parallel='l' in arg, **self.config.kw)
            # self.scenario.update_config(self.config.config)
            self.replace_scenario()
            traces2 = self.run(*a, **kw)
        self.report()
        print("trace1 contains trace2?", traces1.contains(traces2))
        print("trace2 contains trace1?", traces2.contains(traces1))
        return traces1, traces2

    def replace_scenario(self, new_scenario):
        arg = self.config.rest[0]
        self.config.config = ScenarioConfig(
            incremental="i" in arg, parallel="l" in arg, **self.config.kw
        )
        # self.scenario.cleanup_cache()
        self.scenario = new_scenario
        self.scenario.update_config(self.config.config)

    def report(self):
        print("report:")
        print("#agents:", self.num_agent)
        print("agent type:", self.agent_type)
        print("map name:", self.map_name)
        print("postCont:", self.cont_engine)
        print("noisy:", self.noisy_s)
        print("#nodes:", self.num_nodes)
        print("#leaves:", self.leaves)
        print(f"run time: {self.run_time:.2f}s")
        print(f"timesteps: {self.timesteps}s")
        if self.config.config.parallel:
            print(f"parallelness: {self.parallelness:.2f}")
        if self.config.config.incremental:
            print(f"cache size: {self.cache_size:.2f}MB")
            print(f"cache hit: {(self.cache_hits[0], self.cache_hits[1])}")
            print(
                f"cache hit rate: {self.cache_hits[0] / (self.cache_hits[0] + self.cache_hits[1]) * 100:.2f}%"
            )

    def swap_dl(self, id: str, alt_ctlr: str):
        self.scenario.agent_dict[id].decision_logic = ControllerIR.parse(fn=alt_ctlr)
