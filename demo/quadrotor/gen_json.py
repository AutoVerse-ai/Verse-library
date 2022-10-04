from json import dump, dumps, load, loads
import os
import copy
import queue
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.plotter.plotter2D import *

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
total_id = 0


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
    if isinstance(root, AnalysisTree):
        root = root.root
    global curr_lines, tree_dict, total_id
    rst_dict = {'node_name': None, 'id': 0, 'start_line': 1, 'parent': None, 'child': {}, 'agent': {}, 'init': {}, 'mode': {}, 'static': {}, 'start_time': 0,
                'trace': {}, 'type': 'simtrace', 'assert_hits': None}
    if level not in tree_dict:
        tree_dict[level] = id
    rst_dict['node_name'] = f'{level}-{id}'
    i = 0
    if parent_dict == None:
        total_id = 0
        curr_lines = 2
        rst_dict['parent'] = None
    else:
        # i = 1
        rst_dict['parent'] = {
            'name': parent_dict['node_name'], 'index': parent_dict['id'], 'start_line': parent_dict['start_line']}
        # json_obj = dumps(rst_dict, indent=4, separators=(', ', ': '))
        # curr_lines = curr_lines+2+json_obj.count('\n')
        parent_dict['child'][rst_dict['node_name']] = {
            'name': rst_dict['node_name'], 'index': total_id, 'start_line': curr_lines}

    rst_dict['start_line'] = curr_lines
    rst_dict['id'] = total_id
    total_id += 1

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
        i += 1
    json_obj = dumps(rst_dict, indent=4, separators=(', ', ': '))
    curr_lines = curr_lines+json_obj.count('\n')+1+4*i
    child_list = []
    for child in root.child:
        child_list.append(trans_dict(child, level+1, init_id, rst_dict))
        init_id += 1

    json_root = json_node(rst_dict)
    json_root.set_children(child_list)
    return json_root


def write_json(root, file):
    root = trans_dict(root, 0, 0, None)
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


def parse_trace(root, agend_id):
    pass


def read_json(file):
    if not os.path.exists(file):
        print('file not exists')
    with open(file, 'r') as f:
        total_dict = load(f)

    def create_node(index):
        node_dict = total_dict[str(index)]
        mode_dict = {agent_id: loads(mode_json)
                     for agent_id, mode_json in node_dict['mode'].items()}
        init_dict = {agent_id: loads(init_json)
                     for agent_id, init_json in node_dict['init'].items()}
        trace_dict = {}
        for agent_id, trace_json_list in node_dict['trace'].items():
            trace_list = [loads(trace_json) for trace_json in trace_json_list]
            trace_dict[agent_id] = trace_list
        child_list = []
        for child in node_dict['child']:
            child_id = node_dict['child'][child]['index']
            child_list.append(create_node(child_id))

        node = AnalysisTreeNode(
            trace=trace_dict,
            init=init_dict,
            mode=mode_dict,
            static=node_dict['static'],
            agent=node_dict['agent'],
            child=child_list,
            start_time=node_dict['start_time'],
            type='simtrace'
        )
        return node

    root = create_node(0)

    return root


if __name__ == "__main__":
    path = os.path.abspath('.')
    if os.path.exists(path+'/demo'):
        path += '/demo'
    path += '/output'
    file = path+"/output.json"
    root = read_json(file)
