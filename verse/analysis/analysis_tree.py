from enum import Enum, auto
from functools import reduce
from typing import Any, Iterable, List, Dict, Optional, Sequence, TypeVar, Literal
import json
import numpy.typing as nptyp, numpy as np, portion
import networkx as nx
from matplotlib import colors
import matplotlib.pyplot as plt
import graphviz

from verse.analysis.dryvr import _EPSILON
from verse.agents.base_agent import BaseAgent

TraceType = nptyp.NDArray[np.float_]

_T = TypeVar("_T")


def index_of(l: Iterable[_T], item: _T) -> Optional[int]:
    """Get the index of a item in an iterable. Returns None if not found"""
    for i, v in enumerate(l):
        if v == item:
            return i
    return None


class AnalysisTreeNodeType(Enum):
    """Denotes different types of AnalysisTreeNode's. Can either be simulation traces or reach
    tubes."""

    SIM_TRACE = auto()
    REACH_TUBE = auto()

    def __str__(self) -> str:
        return "simtrace" if self == AnalysisTreeNodeType.SIM_TRACE else "reachtube"

    @staticmethod
    def _from_str(s: str) -> "AnalysisTreeNodeType":
        if s == "simtrace":
            return AnalysisTreeNodeType.SIM_TRACE
        if s == "reachtube":
            return AnalysisTreeNodeType.REACH_TUBE
        raise ValueError(f"Invalid string to AnalysisTreeNodeType: {s}")


class AnalysisTreeNode:
    """A AnalysisTreeNode stores the continous execution of the system without transition
    happening"""

    trace: Dict[str, TraceType]
    """The trace for each agent.
    The key of the dict is the agent id and the value of the dict is the simulated traces for each
    agent"""
    init: Dict[str, Sequence[float]]
    """Initial conditions per agent for this node.
    The key of the dict is the agent id and the value of the dict is the range/set of initial
    conditions for that agent"""
    mode: Dict[str, Sequence[str]]
    """Discrete mode per agent for this node.
    The key of the dict is the agent id and the value of the dict is the discrete mode for that
    agent"""
    agent: Dict[str, BaseAgent]
    """Discrete mode per agent for this node.
    The key of the dict is the agent id and the value of the dict is the object instance for that
    agent"""
    height: int
    """The height/depth of the current node in the AnalysisTree"""
    assert_hits: Dict[str, Sequence[str]]  # TODO type
    """Assert hits per agent for this node.
    The key of the dict is the agent id and the value of the dict is the list of labels hit in the
    node for that agent"""
    child: List["AnalysisTreeNode"]
    """A list of children nodes for the current node. Can be empty."""
    start_time: float
    """The earliest simulation time of this node."""
    type: AnalysisTreeNodeType
    """Type of the node. Can either be "simtrace" or "reachtube"."""
    ndigits: int
    """Number of digits to round `start_time` to."""
    static: Dict[str, Sequence[str]]
    """Static data per agent for this node.
    The key of the dict is the agent id and the value of the dict is the static data for that
    agent"""
    uncertain_param: Dict[str, Sequence[str]]
    """Parameters for uncertainty per agent for this node.
    The key of the dict is the agent id and the value of the dict is the uncertainty data for that
    agent"""
    id: int
    """Integer ID for the current node. Unique amongst all nodes in the AnalysisTree"""

    def __init__(
        self,
        trace: Dict[str, TraceType],
        init: Dict[str, Sequence[float]],
        mode: Dict[str, Sequence[str]],
        static: Dict[str, Sequence[str]],
        uncertain_param: Dict[str, Sequence[str]],
        agent: Dict[str, BaseAgent],
        height: int,
        assert_hits: Dict[str, Sequence[str]],
        child: List["AnalysisTreeNode"],
        start_time: float,
        ndigits: int,
        type: AnalysisTreeNodeType,
        id: int,
    ) -> None:
        self.trace = trace
        self.init = init
        self.mode = mode
        self.agent = agent
        self.height = height
        self.child = child
        self.start_time = round(start_time, ndigits)
        self.assert_hits = assert_hits
        self.type = type
        self.static = static
        self.uncertain_param = uncertain_param
        self.id = id
        self.ndigits = ndigits

    @staticmethod
    def root_from_inits(
        init: Dict[str, Sequence[float]],
        mode: Dict[str, Sequence[str]],
        static: Dict[str, Sequence[str]],
        uncertain_param: Dict[str, Sequence[str]],
        agent: Dict[str, BaseAgent],
        ndigits: int,
        type: AnalysisTreeNodeType,
    ) -> "AnalysisTreeNode":
        """Construct the root node from initial conditions and other settings."""
        return AnalysisTreeNode(
            init=init,
            mode=mode,
            static=static,
            uncertain_param=uncertain_param,
            agent=agent,
            ndigits=ndigits,
            type=type,
            trace={},
            height=0,
            child=[],
            assert_hits={},
            start_time=0,
            id=0,
        )

    def new_child(
        self,
        init: Dict[str, Sequence[float]],
        mode: Dict[str, Sequence[str]],
        trace: Dict[str, TraceType],
        start_time: float,
        id: int,
    ) -> "AnalysisTreeNode":
        """Construct a child node, copying unchanged items from the parent node (self)."""
        return AnalysisTreeNode(
            init=init,
            mode=mode,
            trace=trace,
            static=self.static,
            uncertain_param=self.uncertain_param,
            agent=self.agent,
            ndigits=self.ndigits,
            type=self.type,
            height=self.height + 1,
            child=[],
            assert_hits={},
            start_time=start_time,
            id=id,
        )

    def _to_dict(self) -> Dict[str, Any]:
        rst_dict = {
            "id": self.id,
            "parent": None,
            "child": [],
            "agent": {},
            "init": {aid: list(init) for aid, init in self.init.items()},
            "mode": self.mode,
            "height": self.height,
            "static": self.static,
            "start_time": self.start_time,
            "trace": (
                {aid: t.tolist() for aid, t in self.trace.items()}
                if self.type == AnalysisTreeNodeType.SIM_TRACE
                else self.trace
            ),
            "type": str(self.type),
            "assert_hits": self.assert_hits,
            "uncertain_param": self.uncertain_param,
            "ndigits": self.ndigits,
        }
        agent_dict = {}
        for agent_id in self.agent:
            agent_dict[agent_id] = f"{type(self.agent[agent_id])}"
        rst_dict["agent"] = agent_dict

        return rst_dict

    def get_track(self, agent_id: str, mode: Sequence[str]) -> Optional[str]:
        """Filter out the track mode(s) for a given agent"""
        state_defs = self.agent[agent_id].decision_logic.state_defs
        mode_def_names = next(iter(state_defs.values())).disc_type
        track_mode_ind = index_of(mode_def_names, "TrackMode")
        if track_mode_ind is None:
            return None
        return mode[track_mode_ind]

    def get_mode(self, agent_id: str, mode: Sequence[str]) -> Optional[Sequence[str]]:
        """Filter out the agent mode(s) for a given agent"""
        state_defs = self.agent[agent_id].decision_logic.state_defs
        mode_def_names = next(iter(state_defs.values())).disc_type
        track_mode_ind = index_of(mode_def_names, "TrackMode")
        if track_mode_ind is None:
            if len(mode) == 1:
                return mode[0]
            return mode
        if len(mode_def_names) == 2:
            return mode[1 - track_mode_ind]
        return tuple(v for i, v in enumerate(mode) if i != track_mode_ind)

    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> "AnalysisTreeNode":
        return AnalysisTreeNode(
            trace=(
                {aid: np.array(data["trace"][aid]) for aid in data["agent"].keys()}
                if data["type"] == "simtrace"
                else data["trace"]
            ),
            id=data["id"],
            init=data["init"],
            mode=data["mode"],
            height=data["height"],
            static=data["static"],
            agent=data["agent"],
            assert_hits=data["assert_hits"],
            child=[],
            start_time=data["start_time"],
            type=AnalysisTreeNodeType._from_str(data["type"]),
            uncertain_param=data["uncertain_param"],
            ndigits=data["ndigits"],
        )


def _color_interp(c1: str, c2: str, mix: float) -> str:
    return colors.to_hex(
        (1 - mix) * np.array(colors.to_rgb(c1)) + mix * np.array(colors.to_rgb(c2))
    )


class AnalysisTree:
    """A tree containing the reachable states the scenario produced."""

    root: AnalysisTreeNode
    """Root node for the tree"""
    nodes: List[AnalysisTreeNode]
    """All nodes in the tree. Order is not guaranteed"""
    type: AnalysisTreeNodeType
    """Type of the analysis tree"""

    def __init__(self, root: AnalysisTreeNode) -> None:
        self.root = root
        self.nodes = self._get_all_nodes(root)
        self.type = root.type

    @staticmethod
    def _get_all_nodes(root: AnalysisTreeNode) -> List[AnalysisTreeNode]:
        # Perform BFS/DFS to store all the tree node in a list
        res = []
        queue = [root]
        node_id = 0
        while queue:
            node = queue.pop(0)
            res.append(node)
            node_id += 1
            queue += node.child
        return res

    def dump(self, fn: str) -> None:
        """Dumps the AnalysisTree as JSON data to the file "fn"."""
        res_dict = {}
        converted_node = self.root._to_dict()
        res_dict[self.root.id] = converted_node
        queue = [self.root]
        while queue:
            parent_node = queue.pop(0)
            for child_node in parent_node.child:
                node_dict = child_node._to_dict()
                node_dict["parent"] = parent_node.id
                res_dict[child_node.id] = node_dict
                res_dict[parent_node.id]["child"].append(child_node.id)
                queue.append(child_node)

        with open(fn, "w+") as f:
            json.dump(res_dict, f, indent=4, sort_keys=True)

    @staticmethod
    def load(fn: str) -> "AnalysisTree":
        """Loads the AnalysisTree from the file "fn" as JSON data."""
        with open(fn, "r") as f:
            data = json.load(f)
        root_node_dict = data[str(0)]
        root = AnalysisTreeNode._from_dict(root_node_dict)
        queue = [(root_node_dict, root)]
        while queue:
            parent_node_dict, parent_node = queue.pop(0)
            for child_node_idx in parent_node_dict["child"]:
                child_node_dict = data[str(child_node_idx)]
                child_node = AnalysisTreeNode._from_dict(child_node_dict)
                parent_node.child.append(child_node)
                queue.append((child_node_dict, child_node))
        return AnalysisTree(root)

    # TODO Generalize to different timesteps
    def contains(
        self, other: "AnalysisTree", strict: bool = True, tol: Optional[float] = None
    ) -> bool:
        """Checks whether this AnalysisTree constains the `other` AnalysisTree. Returns, for
        reachability, whether the current tree (bloated by a small value) fully contains the other
        tree or not; for simulation, whether the other tree is close enough to the current tree.
        strict: requires set of agents to be the same
        """
        tol = _EPSILON if tol is None else tol
        cur_agents = set(self.nodes[0].agent.keys())
        other_agents = set(other.nodes[0].agent.keys())
        min_agents = list(other_agents)
        types = list(set(n.type for n in self.nodes + other.nodes))
        assert len(types) == 1, f"Different types of nodes: {types}"
        if not (
            (strict and cur_agents == other_agents)
            or (not strict and cur_agents.issuperset(other_agents))
        ):
            return False
        if types[0] == "simtrace":  # Simulation
            if len(self.nodes) != len(other.nodes):
                return False

            def sim_seg_contains(a: Dict[str, TraceType], b: Dict[str, TraceType]) -> bool:
                return all(
                    a[aid].shape == b[aid].shape
                    and bool(np.all(np.abs(a[aid][:, 1:] - b[aid][:, 1:]) < tol))
                    for aid in min_agents
                )

            def sim_node_contains(a: AnalysisTreeNode, b: AnalysisTreeNode) -> bool:
                if not sim_seg_contains(a.trace, b.trace):
                    return False
                if len(a.child) != len(b.child):
                    return False
                child_num = len(a.child)
                other_not_paired = set(range(child_num))
                for i in range(child_num):
                    for j in other_not_paired:
                        if sim_node_contains(a.child[i], b.child[j]):
                            other_not_paired.remove(j)
                            break
                    else:
                        return False
                return True

            return sim_node_contains(self.root, other.root)
        else:  # Reachability
            cont_num = len(other.nodes[0].trace[min_agents[0]][0])

            def collect_ranges(
                n: AnalysisTreeNode,
            ) -> Dict[str, List[List[portion.Interval]]]:
                trace_len = len(n.trace[min_agents[0]])
                cur = {
                    aid: [
                        [
                            portion.closed(n.trace[aid][i][j], n.trace[aid][i + 1][j])
                            for j in range(cont_num)
                        ]
                        for i in range(0, trace_len, 2)
                    ]
                    for aid in min_agents
                }
                if len(n.child) == 0:
                    return {aid: cur[aid] for aid in other_agents}
                else:
                    children = [collect_ranges(c) for c in n.child]
                    child_num = len(children)
                    time_step = round(
                        n.trace[min_agents[0]][-1][0] - n.trace[min_agents[0]][-2][0],
                        10,
                    )
                    start_time_list = [c.start_time for c in n.child]
                    min_start_time = min(start_time_list)
                    bias_list = [
                        round((c.start_time - min_start_time) / time_step) for c in n.child
                    ]
                    trace_len_list = [len(children[i][min_agents[0]]) for i in range(child_num)]
                    trace_len = max(trace_len_list)
                    combined = {
                        aid: [
                            [
                                reduce(
                                    portion.Interval.union,
                                    (
                                        children[k][aid][i - bias_list[k]][j]
                                        if i - bias_list[k] >= 0
                                        and i - bias_list[k] < trace_len_list[k]
                                        else portion.empty()
                                        for k in range(child_num)
                                    ),
                                )
                                for j in range(cont_num)
                            ]
                            for i in range(trace_len)
                        ]
                        for aid in other_agents
                    }
                    overlap_len = round(
                        (n.trace[min_agents[0]][-1][0] - min_start_time) / time_step
                    )
                    overlap = {
                        aid: [
                            [
                                portion.Interval.union(
                                    cur[aid][i][j], combined[aid][overlap_len + i][j]
                                )
                                for j in range(cont_num)
                            ]
                            for i in range(-overlap_len, 0)
                        ]
                        for aid in other_agents
                    }
                    return {
                        aid: cur[aid][:-overlap_len] + overlap[aid] + combined[aid][overlap_len:]
                        for aid in other_agents
                    }

            this_tree, other_tree = collect_ranges(self.root), collect_ranges(other.root)
            total_len = len(other_tree[min_agents[0]])
            assert total_len <= len(this_tree[min_agents[0]])
            # bloat and containment
            return all(
                other_tree[aid][i][j]
                in this_tree[aid][i][j].apply(
                    lambda x: x.replace(lower=lambda v: v - tol, upper=lambda v: v + tol)
                )
                for aid in other_agents
                for i in range(total_len)
                for j in range(cont_num)
            )

    @staticmethod
    def _get_len(node: AnalysisTreeNode, lens: Dict[int, int]) -> int:
        res = len(next(iter(node.trace.values()))) + (
            0 if len(node.child) == 0 else max(AnalysisTree._get_len(c, lens) for c in node.child)
        )
        lens[node.id] = res
        return res

    def visualize(self) -> None:
        """Visualizes the AnalysisTree as a tree graph using `networkx`.
        Each node in the graph will correspond to one AnalysisTreeNode, and the edges denote
        parent/child relationship. Each node will be colored according to a gradient from red to
        blue using the start_time of each node, where red signifies newer nodes and blue signifies
        older nodes.
        """
        lens = {}
        total_len = AnalysisTree._get_len(self.root, lens)
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node.id, time=(node.id, node.start_time))
            for child in node.child:
                G.add_node(child.id, time=(child.id, child.start_time))
                G.add_edge(node.id, child.id)
        labels = nx.get_node_attributes(G, "time")
        nx.draw_planar(
            G,
            node_color=[_color_interp("red", "blue", lens[id] / total_len) for id in G],
            labels=labels,
        )
        plt.show()

    def visualize_dot(
        self,
        filename: str,
        otype: Literal["png", "svg", "pdf", "jpg"] = "png",
        font: Optional[str] = None,
    ) -> None:
        """Same as `visualize`, but rather use `graphviz` to for visualizing the tree.
        `filename` is the prefix, i.e. doesn't include extensions. `filename.dot` will be saved as
        well as `filename.png`
        """

        def diff(a: AnalysisTreeNode, b: AnalysisTreeNode) -> List[str]:
            return [aid for aid in a.agent if a.mode[aid] != b.mode[aid]]

        lens = {}
        total_len = AnalysisTree._get_len(self.root, lens)
        graph = graphviz.Digraph()
        for node in self.nodes:
            tooltip = "\n".join(
                f"{aid}: {[*node.mode[aid], *node.init[aid]]}" for aid in node.agent
            )
            graph.node(
                str(node.id),
                label=str(node.id),
                color=_color_interp("red", "blue", lens[node.id] / total_len),
                tooltip=tooltip,
            )
            for c in node.child:
                d = diff(node, c)
                tooltip = "\n".join(f"{aid}: {node.mode[aid]} -> {c.mode[aid]}" for aid in d)
                graph.edge(str(node.id), str(c.id), label=", ".join(d), tooltip=tooltip)
        if font is not None:
            graph.node_attr.update(fontname=font)
            graph.edge_attr.update(fontname=font)
            graph.graph_attr.update(fontname=font)
        graph.render(
            filename + ".dot",
            format=otype,
            outfile=filename + "." + otype,
            engine="twopi",
        )

    def is_equal(self, other: "AnalysisTree") -> bool:
        """Compares if 2 AnalysisTree's traces are close enough. For simulation, this simply
        compares if the point data are within a small range. For reachability, this checks if the
        start and end points respectively for each dimension are within a small range of the other
        tree"""
        return self.contains(other) and other.contains(self)

    def leaves(self) -> int:
        """Return the number of leaves for this tree"""
        count = 0
        for node in self.nodes:
            if len(node.child) == 0:
                count += 1
        return count


def first_transitions(tree: AnalysisTree) -> Dict[str, float]:  # id, start time
    """Collects the time when each agent first transitions for a AnalysisTree. Returns a dict
    mapping from agent IDs to the simulation time when it first transitions"""
    d = {}
    for node in tree.nodes:
        for child in node.child:
            for aid in node.agent:
                if aid not in d and node.mode[aid] != child.mode[aid]:
                    d[aid] = child.start_time
    return d
