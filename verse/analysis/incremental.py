from collections import defaultdict
from dataclasses import dataclass
from pprint import pp
from typing import Any, DefaultDict, List, Tuple, Optional, Dict
from verse.agents.base_agent import BaseAgent
from verse.analysis import AnalysisTreeNode
from intervaltree import IntervalTree
import itertools, copy, numpy as np

from verse.analysis.dryvr import _EPSILON
# from verse.analysis.simulator import PathDiffs
from verse.parser.parser import ControllerIR, ModePath

@dataclass
class CachedTransition:
    inits: Dict[str, List[float]]
    transition: int
    disc: List[str]
    cont: List[float]
    paths: List[ModePath]

@dataclass
class CachedSegment:
    trace: List[List[float]]
    asserts: List[str]
    transitions: List[CachedTransition]
    run_num: int
    node_id: int

@dataclass
class CachedReachTrans:
    inits: Dict[str, List[float]]
    transition: int
    mode: List[str]
    dest: List[float]
    reset: List[float]
    reset_idx: List[int]
    paths: List[ModePath]

@dataclass
class CachedRTTrans:
    asserts: List[str]
    transitions: List[CachedReachTrans]
    run_num: int
    node_id: int

def to_simulate(old_agents: Dict[str, BaseAgent], new_agents: Dict[str, BaseAgent], cached: Dict[str, CachedSegment]) -> Tuple[Dict[str, CachedSegment], Any]: #s/Any/PathDiffs/
    assert set(old_agents.keys()) == set(new_agents.keys())
    removed_paths, added_paths, reset_changed_paths = [], [], []
    for agent_id, old_agent in old_agents.items():
        new_agent = new_agents[agent_id]
        old_ctlr, new_ctlr = old_agent.decision_logic, new_agent.decision_logic
        assert old_ctlr.args == new_ctlr.args
        def group_by_var(ctlr: ControllerIR) -> Dict[str, List[ModePath]]:
            grouped = defaultdict(list)
            for path in ctlr.paths:
                grouped[path.var].append(path)
            return dict(grouped)
        old_grouped, new_grouped = group_by_var(old_ctlr), group_by_var(new_ctlr)
        if set(old_grouped.keys()) != set(new_grouped.keys()):
            raise NotImplementedError("different variable outputs")
        for var, old_paths in old_grouped.items():
            new_paths = new_grouped[var]
            for i, (old, new) in enumerate(itertools.zip_longest(old_paths, new_paths)):
                if new == None:
                    removed_paths.append((agent_id, old))
                elif not ControllerIR.ir_eq(old.cond_veri, new.cond_veri):
                    # pp(("diff", old, new))
                    # pp(("diff", ControllerIR.dump(old.cond_veri), ControllerIR.dump(old.val_veri), ControllerIR.dump(new.cond_veri), ControllerIR.dump(new.val_veri)))
                    removed_paths.append((agent_id, old))
                    added_paths.append((new_agent, new))
                elif not ControllerIR.ir_eq(old.val_veri, new.val_veri):
                    reset_changed_paths.append(new)
    new_cache = {}
    # pp(("removed_paths", removed_paths))
    for agent_id in cached:
        segment = copy.deepcopy(cached[agent_id])
        new_transitions = []
        for trans in segment.transitions:
            removed = False
            for path in trans.paths:
                for aid, rp in removed_paths:
                    if aid == agent_id and rp == path:
                        # pp(("remove", path, rp))
                        # pp(("remove", ControllerIR.dump(path.cond_veri), ControllerIR.dump(path.val_veri), ControllerIR.dump(rp.cond_veri), ControllerIR.dump(rp.val_veri)))
                        removed = True
                for rcp in reset_changed_paths:
                    if path.cond == rcp.cond:
                        path.val = rcp.val
            # pp(("filter", agent_id, trans.paths, removed))
            if not removed:
                new_transitions.append(trans)
        segment.transitions = new_transitions
        new_cache[agent_id] = segment
        # pp(("filtered", agent_id, len(cached[agent_id].transitions), len(new_cache[agent_id].transitions), len([p for a, p in added_paths if a.id == agent_id])))
    return new_cache, added_paths

def convert_sim_trans(agent_id, transit_agents, inits, transition, trans_ind):
    if agent_id in transit_agents:
        return [CachedTransition(inits, trans_ind, mode, init, paths) for _id, mode, init, paths in transition]
    else:
        return []

def convert_reach_trans(agent_id, transit_agents, inits, transition, trans_ind):
    if agent_id in transit_agents:
        return [CachedReachTrans(inits, trans_ind, mode, dest, reset, reset_idx, paths) for _id, mode, dest, reset, reset_idx, paths in transition]
    else:
        return []

def combine_all(inits):
    return [[min(a) for a in np.transpose(np.array(inits)[:, 0])],
            [max(a) for a in np.transpose(np.array(inits)[:, 1])]]

def sim_trans_suit(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> bool:
    assert set(a.keys()) == set(b.keys())
    return all(abs(av - bv) < _EPSILON for aid in a.keys() for av, bv in zip(a[aid], b[aid]))

def reach_trans_suit(a: Dict[str, List[List[List[float]]]], b: Dict[str, List[List[List[float]]]]) -> bool:
    # FIXME include discrete stuff
    assert set(a.keys()) == set(b.keys())
    def transp(a):
        return list(map(list, zip(*a)))
    def suits(a: List[List[float]], b: List[List[float]]) -> bool:
        at, bt = transp(a), transp(b)
        return all(al <= bl and ah >= bh for (al, ah), (bl, bh) in zip(at, bt))
    return all(suits(av, bv) for aid in a.keys() for av, bv in zip(a[aid], b[aid]))

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

    def add_segment(self, agent_id: str, node: AnalysisTreeNode, transit_agents: List[str], trace: List[List[float]], transition, trans_ind: int, run_num: int):
        key = (agent_id,) + tuple(node.mode[agent_id])
        init = node.init[agent_id]
        tree = self.cache[key]
        assert_hits = node.assert_hits or {}
        # pp(('add seg', agent_id, *node.mode[agent_id], *init))
        for i, val in enumerate(init):
            if i == len(init) - 1:
                transitions = convert_sim_trans(agent_id, transit_agents, node.init, transition, trans_ind)
                entry = CachedSegment(trace, assert_hits.get(agent_id), transitions, run_num, node.id)
                tree[val - _EPSILON:val + _EPSILON] = entry
                return entry
            else:
                next_level_tree = IntervalTree()
                tree[val - _EPSILON:val + _EPSILON] = next_level_tree
                tree = next_level_tree
        raise Exception("???")

    @staticmethod
    def iter_tree(tree, depth: int) -> List[List[float]]:
        if depth == 0:
            return [[(i.begin + i.end) / 2, (i.data.run_num, i.data.node_id, [t.transition for t in i.data.transitions], len(i.data.trace))] for i in tree]
        res = []
        for i in tree:
            mid = (i.begin + i.end) / 2
            subs = SimTraceCache.iter_tree(i.data, depth - 1)
            res.extend([mid] + sub for sub in subs)
        return res

    def get_cached_inits(self, n: int):
        inits = defaultdict(list)
        for key, tree in self.cache.items():
            inits[key[0]].extend((*key[1:], *init) for init in self.iter_tree(tree, n))
        inits = {k: sorted(v) for k, v in inits.items()}
        return inits

    @staticmethod
    def query_cont(tree: IntervalTree, cont: List[float]) -> List[CachedSegment]:
        assert isinstance(tree, IntervalTree)
        next_level_entries = [t.data for t in tree[cont[0]]]
        if len(cont) == 1:
            for ent in next_level_entries:
                assert isinstance(ent, CachedSegment)
            return next_level_entries
        else:
            return [ent for t in next_level_entries for ent in SimTraceCache.query_cont(t, cont[1:])]

    def check_hit(self, agent_id: str, mode: Tuple[str], init: List[float], inits: Dict[str, List[float]]) -> Optional[CachedSegment]:
        key = (agent_id,) + tuple(mode)
        if key not in self.cache:
            return None
        tree = self.cache[key]
        entries = self.query_cont(tree, init)
        if len(entries) == 0:
            return None
        def num_trans_suit(e: CachedSegment) -> int:
            return sum(1 if sim_trans_suit(t.inits, inits) else 0 for t in e.transitions)
        entries = list(sorted([(e, -num_trans_suit(e)) for e in entries], key=lambda p: p[1]))
        # pp(("check hit entries", len(entries), entries[0][1]))
        assert isinstance(entries[0][0], (type(None), CachedSegment))
        return entries[0][0]

class TubeCache:
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

class ReachTubeCache:
    def __init__(self):
        self.cache: DefaultDict[tuple, IntervalTree] = defaultdict(IntervalTree)

    def add_tube(self, agent_id: str, init: Dict[str, List[List[float]]], node: AnalysisTreeNode, transit_agents: List[str], transition, trans_ind: int, run_num: int):
        key = (agent_id,) + tuple(node.mode[agent_id])
        tree = self.cache[key]
        # pp(('add seg', agent_id, node.mode[agent_id], init))
        assert_hits = node.assert_hits or {}
        init = list(map(tuple, zip(*init[agent_id])))
        for i, (low, high) in enumerate(init):
            if i == len(init) - 1:
                transitions = convert_reach_trans(agent_id, transit_agents, node.init, transition, trans_ind)
                entry = CachedRTTrans(assert_hits.get(agent_id), transitions, node.agent[agent_id].decision_logic, run_num, node.id)
                tree[low:high + _EPSILON] = entry
                return entry
            else:
                next_level_tree = IntervalTree()
                tree[low:high + _EPSILON] = next_level_tree
                tree = next_level_tree
        raise Exception("???")

    @staticmethod
    def query_cont(tree: IntervalTree, cont: List[Tuple[float, float]]) -> List[CachedRTTrans]:
        assert isinstance(tree, IntervalTree)
        low, high = cont[0]
        next_level_entries = [t.data for t in tree[low:high + _EPSILON] if t.begin <= low and high <= t.end]
        if len(cont) == 1:
            for ent in next_level_entries:
                assert isinstance(ent, CachedRTTrans)
            return next_level_entries
        else:
            return [ent for t in next_level_entries for ent in ReachTubeCache.query_cont(t, cont[1:])]

    def check_hit(self, agent_id: str, mode: Tuple[str], init: List[float], inits: Dict[str, List[List[List[float]]]]) -> Optional[CachedRTTrans]:
        key = (agent_id,) + tuple(mode)
        if key not in self.cache:
            return None
        tree = self.cache[key]
        entries = self.query_cont(tree, list(map(tuple, zip(*init))))
        if len(entries) == 0:
            return None
        def num_trans_suit(e: CachedRTTrans) -> int:
            return sum(1 if reach_trans_suit(t.inits, inits) else 0 for t in e.transitions)
        entries = list(sorted([(e, -num_trans_suit(e)) for e in entries], key=lambda p: p[1]))
        # pp(("check hit entries", len(entries), entries[0][1]))
        assert isinstance(entries[0][0], (type(None), CachedRTTrans))
        return entries[0][0]

    @staticmethod
    def iter_tree(tree, depth: int) -> List[List[float]]:
        if depth == 0:
            return [[(i.begin + i.end) / 2, (i.data.run_num, i.data.node_id, [t.transition for t in i.data.transitions])] for i in tree]
        res = []
        for i in tree:
            mid = (i.begin + i.end) / 2
            subs = ReachTubeCache.iter_tree(i.data, depth - 1)
            res.extend([mid] + sub for sub in subs)
        return res

    def get_cached_inits(self, n: int):
        inits = defaultdict(list)
        for key, tree in self.cache.items():
            inits[key[0]].extend((*key[1:], *init) for init in self.iter_tree(tree, n))
        inits = dict(inits)
        return inits
