from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Tuple, Optional
from verse.analysis import AnalysisTreeNode
from intervaltree import IntervalTree

from verse.analysis.dryvr import _EPSILON
from verse.parser.parser import ControllerIR

@dataclass
class CachedTransition:
    transition: int
    disc: List[str]
    cont: List[float]

@dataclass
class CachedSegment:
    trace: List[List[float]]
    asserts: List[str]
    transitions: List[CachedTransition]
    controller: ControllerIR

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

    def add_segment(self, agent_id: str, node: AnalysisTreeNode):
        key = (agent_id,) + tuple(node.mode[agent_id])
        init = node.init[agent_id]
        tree = self.cache[key]
        for i, val in enumerate(init):
            if i == len(init) - 1:
                transitions = [CachedTransition(len(n.trace[agent_id]), n.mode[agent_id], n.init[agent_id]) for n in node.child]
                entry = CachedSegment(node.trace[agent_id], node.assert_hits.get(agent_id), transitions, node.agent[agent_id].controller)
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
