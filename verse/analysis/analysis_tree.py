from typing import List, Dict, Any
import json
from treelib import Tree
import numpy as np

class AnalysisTreeNode:
    """AnalysisTreeNode class
    A AnalysisTreeNode stores the continous execution of the system without transition happening"""
    trace: Dict
    """The trace for each agent. 
    The key of the dict is the agent id and the value of the dict is simulated traces for each agent"""
    init: Dict 
    
    def __init__(
        self,
        trace={},
        init={},
        mode={},
        static = {},
        uncertain_param = {},
        agent={},
        assert_hits={},
        child=[],
        start_time = 0,
        ndigits = 10,
        type = 'simtrace',
        id = 0
    ):
        self.trace:Dict = trace
        self.init: Dict[str, List[float]] = init
        self.mode: Dict[str, List[str]] = mode
        self.agent: Dict = agent
        self.child: List[AnalysisTreeNode] = child
        self.start_time: float = round(start_time, ndigits)
        self.assert_hits = assert_hits
        self.type: str = type
        self.static: Dict[str, List[str]] = static
        self.uncertain_param: Dict[str, List[str]] = uncertain_param
        self.id: int = id

    def to_dict(self):
        rst_dict = {
            'id': self.id, 
            'parent': None, 
            'child': [], 
            'agent': {}, 
            'init': self.init, 
            'mode': self.mode, 
            'static': self.static, 
            'start_time': self.start_time,
            'trace': self.trace, 
            'type': self.type, 
            'assert_hits': self.assert_hits
        }
        agent_dict = {}
        for agent_id in self.agent:
            agent_dict[agent_id] = f'{type(self.agent[agent_id])}'
        rst_dict['agent'] = agent_dict

        return rst_dict

    def get_track(self, agent_id, D):
        if 'TrackMode' not in self.agent[agent_id].decision_logic.mode_defs:
            return ""
        for d in D:
            if d in self.agent[agent_id].decision_logic.mode_defs['TrackMode'].modes:
                return d
        return ""

    def get_mode(self, agent_id, D):
        res = []
        if 'TrackMode' not in self.agent[agent_id].decision_logic.mode_defs:
            if len(D)==1:
                return D[0]
            return D
        for d in D:
            if d not in self.agent[agent_id].decision_logic.mode_defs['TrackMode'].modes:
                res.append(d)
        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)
            
    @staticmethod
    def from_dict(data) -> "AnalysisTreeNode":
        return AnalysisTreeNode(
            trace = data['trace'],
            init = data['init'],
            mode = data['mode'],
            static = data['static'],
            agent = data['agent'],
            assert_hits = data['assert_hits'],
            child = [],
            start_time = data['start_time'],
            type = data['type'],
        )

    def quick_check(self, other: "AnalysisTreeNode", atol = 1e-5) -> bool:
        if not (
                # self.init==other.init and 
                self.mode==other.mode and 
                self.agent==other.agent and 
                self.start_time==other.start_time and 
                self.assert_hits==other.assert_hits and 
                self.type==other.type and 
                # self.static==other.static and 
                # self.uncertain_param==other.uncertain_param and
                len(self.child) == len(other.child) and
                True
                #  self.id==other.id
                 ):
            return False
        # for agent in self.agent:
        #     if not np.allclose(trace, trace_other, 0, atol, equal_nan=True):
        #         return False            
        return True 

    def check_inclusion(self, other: "AnalysisTreeNode", atol = 1e-5) -> bool:
        # for simtrace, return True iff self and other are the same with the error toleration
        # for reachtube, return True iff the trace of self contains trace of other with the same structure and error toleration
        if not self.quick_check(other, atol):
            return False
        # atol = pow(10, -num_digits)
        if self.type=='simtrace':
            for agent, trace in self.trace.items():
                trace_other = other.trace[agent]
                assert trace.shape == trace_other.shape
                # absolute(a - b) <= (atol + rtol * absolute(b))
                if not np.allclose(trace, trace_other, 0, atol, equal_nan=True):
                    return False
        elif self.type=='reachtube':
            for agent, trace in self.trace.items():
                # trace = np.around(trace, num_digits)
                # trace_other = np.around(other.trace[agent], num_digits)
                assert trace.shape == trace_other.shape
                # when considering error, a - b >= -error is trusted as a>=b
                ge_other = np.subtract(trace, trace_other) + atol >= 0
                le_other = np.subtract(trace, trace_other) - atol <= 0
                if not ((np.all(le_other[0::2]) == False) and (np.all(ge_other[1::2]) == True)):
                    return False 
        else:
            raise ValueError
        child_match = 0
        for child in self.child:
            for child_other in other.child:
               if child.check_inclusion(child_other):
                    child_match+=1
                    break
        if child_match == len(self.child):
            return True
        return False


class AnalysisTree:
    def __init__(self, root):
        self.root:AnalysisTreeNode = root
        self.nodes:List[AnalysisTreeNode] = self.get_all_nodes(root)

    def get_all_nodes(self, root: AnalysisTreeNode) -> List[AnalysisTreeNode]:
        # Perform BFS/DFS to store all the tree node in a list
        res = []
        queue = [root]
        node_id = 0
        while queue:
            node = queue.pop(0)
            node.id = node_id 
            res.append(node)
            node_id += 1
            queue += node.child
        return res

    def dump(self, fn):
        res_dict = {}
        converted_node = self.root.to_dict()
        res_dict[self.root.id] = converted_node
        queue = [self.root]
        while queue:
            parent_node = queue.pop(0)
            for child_node in parent_node.child:
                node_dict = child_node.to_dict()
                node_dict['parent'] = parent_node.id
                res_dict[child_node.id] = node_dict 
                res_dict[parent_node.id]['child'].append(child_node.id)
                queue.append(child_node)

        with open(fn,'w+') as f:           
            json.dump(res_dict,f, indent=4, sort_keys=True)

    @staticmethod 
    def load(fn):
        f = open(fn, 'r')
        data = json.load(f)
        f.close()
        root_node_dict = data[str(0)]
        root = AnalysisTreeNode.from_dict(root_node_dict)
        queue = [(root_node_dict, root)]
        while queue:
            parent_node_dict, parent_node = queue.pop(0)
            for child_node_idx in parent_node_dict['child']:
                child_node_dict = data[str(child_node_idx)]
                child_node = AnalysisTreeNode.from_dict(child_node_dict)
                parent_node.child.append(child_node)
                queue.append((child_node_dict, child_node))
        return AnalysisTree(root)

    def dump_tree(self):
        tree = Tree()
        AnalysisTree._dump_tree(self.root, tree, 0, 1)
        tree.show()

    @staticmethod
    def _dump_tree(node, tree, pid, id):
        n = "\n".join(str((aid, *node.mode[aid], *node.init[aid])) for aid in node.agent)
        if pid != 0:
            tree.create_node(n, id, parent=pid)
        else:
            tree.create_node(n, id)
        nid = id + 1
        for child in node.child:
            nid = AnalysisTree._dump_tree(child, tree, id, nid)
        return nid + 1


    def __eq__(self, other: object) -> bool:
        assert isinstance(other, AnalysisTree)
        if len(self.nodes) != len(other.nodes):
            return False
        return self.root.check_inclusion(other.root)

