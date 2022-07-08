from json import dump, dumps, load, loads
import os
import copy
import queue
from dryvr_plus_plus.scene_verifier.analysis.analysis_tree_node import AnalysisTreeNode

AGENT = 'agent'
INIT = 'init'
MODE = 'mode'
STATIC = 'static'
START_TIME = 'start_time'
CHILD = 'child'
TRACE = 'trace'
TYPE = 'simtrace'
ASSERT_HITS = 'assert_hits'

curr_lines = 2
tree_dict = {}


class json_node():
    def __init__(self, data) -> None:
        self.data = data
        self.child = []

    def add_child(self, child):
        self.child.append(child)

    def set_children(self, children):
        self.child = children


def write_json(root, file):
    with open(file, 'w', encoding='utf-8') as f:
        dump(root, f)
    return


def trans_dict(root: AnalysisTreeNode, level, id, parent_dict):
    global curr_lines, tree_dict
    rst_dict = {'node_name': None, 'parent': None, 'agent': {}, 'init': {}, 'mode': {}, 'static': {}, 'start_time': 0,
                'child': {}, 'trace': {}, 'type': 'simtrace', 'assert_hits': None}
    if level not in tree_dict:
        tree_dict[level] = id
    rst_dict['node_name'] = f'{level}-{id}'
    if parent_dict == None:
        rst_dict['parent'] = None
    else:
        rst_dict['parent'] = parent_dict['node_name']
        # json_obj = dumps(rst_dict, indent=4, separators=(', ', ': '))
        # curr_lines = curr_lines+2+json_obj.count('\n')
        parent_dict['child'][rst_dict['node_name']] = curr_lines

    agent_dict = {}
    for agent_id in root.agent:
        agent_dict[agent_id] = f'{type(root.agent[agent_id])}'
    rst_dict['agent'] = agent_dict

    init_dict = {}
    for agent_id in root.init:
        init_dict[agent_id] = dumps(root.init[agent_id])
    rst_dict['init'] = init_dict

    mode_dict = {}
    for agent_id in root.mode:
        mode_dict[agent_id] = dumps(root.mode[agent_id])
    rst_dict['mode'] = mode_dict

    static_dict = {}
    for agent_id in root.static:
        static_dict[agent_id] = dumps(root.static[agent_id])
    rst_dict['static'] = static_dict

    rst_dict['start_time'] = root.start_time
    rst_dict['type'] = root.type

    trace_dict = {}
    for agent_id in root.trace:
        trace_dict[agent_id] = [dumps(step) for step in root.trace[agent_id]]
    rst_dict['trace'] = trace_dict

    if level+1 not in tree_dict:
        tree_dict[level+1] = 0
    init_id = tree_dict[level+1]
    for child in root.child:
        rst_dict['child'][f'{level+1}-{tree_dict[level+1]}'] = None
        tree_dict[level+1] += 1
    json_obj = dumps(rst_dict, indent=4, separators=(', ', ': '))
    curr_lines = curr_lines+json_obj.count('\n')+1
    child_list = []
    for child in root.child:
        child_list.append(trans_dict(child, level+1, init_id, rst_dict))
        init_id += 1


# =========
    # json_obj = dumps(rst_dict, indent=4, separators=(', ', ': '))
    # print(json_obj)

    json_root = json_node(rst_dict)
    json_root.set_children(child_list)
    return json_root


def print_trace(root):
    path = os.path.abspath('.')
    if os.path.exists(path+'/demo'):
        path += '/demo'
    path += '/output'
    if not os.path.exists(path):
        os.makedirs(path)
    file = path+"/output.json"
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'a') as f:
        f.write('{\n')

        def output(root, cnt):
            f.write(f'"{cnt}": ')
            cnt += 1
            f.write(dumps(root.data, indent=4, separators=(', ', ': ')))
            for child in root.child:
                f.write(',\n')
                cnt = output(child, cnt)
            return cnt
        output(root, 0)
        f.write('\n}')
