from collections import defaultdict
from dataclasses import dataclass
from pprint import pp
from typing import DefaultDict, Dict, List, Tuple, Optional
from verse.analysis import AnalysisTreeNode
from intervaltree import IntervalTree

from verse.analysis.dryvr import _EPSILON
from verse.parser.parser import ControllerIR, ModePath

@dataclass
class CachedTransition:
    transition: int
    disc: List[str]
    cont: List[float]
    paths: List[ModePath]

@dataclass
class CachedSegment:
    trace: List[List[float]]
    asserts: List[str]
    transitions: List[CachedTransition]
    controller: ControllerIR
    run_num: int
    node_id: int

@dataclass
class CachedTube:
    tube: List[List[List[float]]]

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return (self.tube == other.tube).any()

class SimTraceCache:
    def __init__(self):
        self.cache: DefaultDict[tuple, IntervalTree] = defaultdict(IntervalTree)

    def add_segment(self, agent_id: str, node: AnalysisTreeNode, trace: List[List[float]], transition_paths: List[List[ModePath]], run_num: int):
        assert len(transition_paths) == len(node.child)
        key = (agent_id,) + tuple(node.mode[agent_id])
        init = node.init[agent_id]
        tree = self.cache[key]
        assert_hits = node.assert_hits or {}
        for i, val in enumerate(init):
            if i == len(init) - 1:
                transitions = [CachedTransition(len(node.trace[agent_id]), n.mode[agent_id], n.init[agent_id], p) for n, p in zip(node.child, transition_paths)]
                entry = CachedSegment(trace, assert_hits.get(agent_id), transitions, node.agent[agent_id].controller, run_num, node.id)
                tree[val - _EPSILON:val + _EPSILON] = entry
                return entry
            else:
                next_level_tree = IntervalTree()
                tree[val - _EPSILON:val + _EPSILON] = next_level_tree
                tree = next_level_tree
        raise Exception("???")

    def check_hit(self, agent_id: str, mode: Tuple[str], init: List[float]) -> Optional[CachedSegment]:
        key = (agent_id,) + tuple(mode)
        if key not in self.cache:
            return None
        tree = self.cache[key]
        for cont in init:
            next_level_entries = list(tree[cont])
            if len(next_level_entries) == 0:
                return None
            tree = min(next_level_entries, key=lambda e: (e.end + e.begin) / 2 - cont).data
        assert isinstance(tree, CachedSegment)
        return tree
    
class ReachTraceCache:
    def __init__(self):
        self.cache: DefaultDict[tuple, IntervalTree] = defaultdict(IntervalTree)

    def add_tube(self, agent_id: str, mode: Tuple[str], init: List[List[float]], trace: List[List[List[float]]]):
        key = (agent_id,) + tuple(mode)
        init = list(map(list, zip(*init)))
        tree = self.cache[key]
        for i, (low, high) in enumerate(init):
            if i == len(init) - 1:
                entry = CachedTube(trace)
                tree[low:high + _EPSILON] = entry
                return entry
            else:
                next_level_tree = IntervalTree()
                tree[low:high + _EPSILON] = next_level_tree
                tree = next_level_tree
        raise Exception("???")

    def check_hit(self, agent_id: str, mode: Tuple[str], init: List[List[float]]) -> Optional[CachedTube]:
        key = (agent_id,) + tuple(mode)
        if key not in self.cache:
            return None
        tree = self.cache[key]
        for low, high in list(map(list, zip(*init))):
            next_level_entries = [t for t in tree[low:high + _EPSILON] if t.begin <= low and high <= t.end]
            if len(next_level_entries) == 0:
                return None
            tree = min(next_level_entries, key=lambda e: low - e.begin + e.end - high).data
        assert isinstance(tree, CachedTube)
        return tree
