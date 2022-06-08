import ast
from email import generator
from re import M 
import astunparse
from enum import Enum, auto
import itertools
import copy
from typing import Any, Dict, List, Tuple

class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()

class LaneMode(Enum):
    Lane0 = auto()
    Lane1 = auto()
    Lane2 = auto()

class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()        # Pedestrians
    Sign = auto()       # Signs, stop signs, merge, yield etc.
    Signal = auto()     # Traffic lights
    Obstacle = auto()   # Static (to road/lane) obstacles

class State:
    x: float
    y: float
    theta: float
    v: float
    vehicle_mode: VehicleMode
    lane_mode: LaneMode
    type: LaneObjectMode

    def __init__(self, x: float = 0, y: float = 0, theta: float = 0, v: float = 0, vehicle_mode: VehicleMode = VehicleMode.Normal, lane_mode: LaneMode = LaneMode.Lane0, type: LaneObjectMode = LaneObjectMode.Vehicle):
        pass

def _parse_elt(root, cont_var_dict, disc_var_dict, iter_name_list, targ_name_list, iter_pos_list) -> Any:
    # Loop through all node in the elt ast 
    for node in ast.walk(root):
        # If the node is an attribute
        if isinstance(node, ast.Attribute):
            if node.value.id in targ_name_list:
                # Find corresponding targ_name in the targ_name_list
                targ_name = node.value.id
                var_index = targ_name_list.index(targ_name)

                # Find the corresponding iter_name in the iter_name_list 
                iter_name = iter_name_list[var_index]

                # Create the name for the tmp variable 
                iter_pos = iter_pos_list[var_index]
                tmp_variable_name = f"{iter_name}_{iter_pos}.{node.attr}"

                # Replace variables in the etl by using tmp variables
                root = ValueSubstituter(tmp_variable_name, node).visit(root)

                # Find the value of the tmp variable in the cont/disc_var_dict
                # Add the tmp variables into the cont/disc_var_dict
                variable_name = iter_name + '.' + node.attr
                variable_val = None
                if variable_name in cont_var_dict:
                    variable_val = cont_var_dict[variable_name][iter_pos]
                    cont_var_dict[tmp_variable_name] = variable_val
                elif variable_name in disc_var_dict:
                    variable_val = disc_var_dict[variable_name][iter_pos]
                    disc_var_dict[tmp_variable_name] = variable_val

        if isinstance(node, ast.Name):
            if node.id in targ_name_list:
                node:ast.Name
                # Find corresponding targ_name in the targ_name_list
                targ_name = node.id
                var_index = targ_name_list.index(targ_name)

                # Find the corresponding iter_name in the iter_name_list 
                iter_name = iter_name_list[var_index]

                # Create the name for the tmp variable 
                iter_pos = iter_pos_list[var_index]
                tmp_variable_name = f"{iter_name}_{iter_pos}"

                # Replace variables in the etl by using tmp variables
                root = ValueSubstituter(tmp_variable_name, node).visit(root)

                # Find the value of the tmp variable in the cont/disc_var_dict
                # Add the tmp variables into the cont/disc_var_dict
                variable_name = iter_name
                variable_val = None
                if variable_name in cont_var_dict:
                    variable_val = cont_var_dict[variable_name][iter_pos]
                    cont_var_dict[tmp_variable_name] = variable_val
                elif variable_name in disc_var_dict:
                    variable_val = disc_var_dict[variable_name][iter_pos]
                    disc_var_dict[tmp_variable_name] = variable_val

    # Return the modified node
    return root

class ValueSubstituter(ast.NodeTransformer):
    def __init__(self, val:str, node):
        super().__init__()
        self.val = val
        self.node = node
    
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node == self.node:
            return ast.Name(
                id = self.val, 
                ctx = ast.Load()
            )
        return node

    def visit_Name(self, node: ast.Attribute) -> Any:
        if node == self.node:
            return ast.Name(
                id = self.val,
                ctx = ast.Load
            )
        return node

    def visit_Call(self, node: ast.Call) -> Any:
        if node == self.node:
            if node.func.id == 'any':
                return ast.BoolOp(
                    op = ast.Or(),
                    values = self.val
            )
            elif node.func.id == 'all':
                return ast.BoolOp(
                    op = ast.And(),
                    values = self.val
                )
        return node

def parse_any_all(
    node: ast.Call, 
    cont_var_dict: Dict[str, float], 
    disc_var_dict: Dict[str, float], 
    len_dict: Dict[str, int]
) -> ast.BoolOp:
    parse_arg = node.args[0]
    if isinstance(parse_arg, ast.GeneratorExp):
        iter_name_list = []
        targ_name_list = []
        iter_len_list = []
        # Get all the iter, targets and the length of iter list 
        for generator in parse_arg.generators:
            iter_name_list.append(generator.iter.id) # a_list
            targ_name_list.append(generator.target.id) # a
            iter_len_list.append(range(len_dict[generator.iter.id])) # len(a_list)

        elt = parse_arg.elt
        expand_elt_ast_list = []
        iter_len_list = list(itertools.product(*iter_len_list))
        # Loop through all possible combination of iter value
        for i in range(len(iter_len_list)):
            changed_elt = copy.deepcopy(elt)
            iter_pos_list = iter_len_list[i]
            # substitute temporary variable in each of the elt and add corresponding variables in the variable dicts
            parsed_elt = _parse_elt(changed_elt, cont_var_dict, disc_var_dict, iter_name_list, targ_name_list, iter_pos_list)
            # Add the expanded elt into the list 
            expand_elt_ast_list.append(parsed_elt)
        # Create the new boolop (and/or) node based on the list of expanded elt
        return ValueSubstituter(expand_elt_ast_list, node).visit(node)
    else:
        return node

class NodeSubstituter(ast.NodeTransformer):
    def __init__(self, old_node, new_node):
        super().__init__()
        self.old_node = old_node 
        self.new_node = new_node

    def visit_Call(self, node: ast.Call) -> Any:
        if node == self.old_node:
            return self.new_node 
        else:
            return node

def unroll_any_all(root, cont_var_dict: Dict[str, float], disc_var_dict: Dict[str, float], len_dict: Dict[str, int]) -> None: 
    i = 0
    while i < sum(1 for _ in ast.walk(root)):
        # TODO: Find a faster way to access nodes in the tree
        node = list(ast.walk(root))[i]
        if isinstance(node, ast.Call) and\
            isinstance(node.func, ast.Name) and\
            (node.func.id=='any' or node.func.id=='all'):
            new_node = parse_any_all(node, cont_var_dict, disc_var_dict, len_dict)
            root = NodeSubstituter(node, new_node).visit(root)
        i += 1
    return root 

if __name__ == "__main__":
    others = [State(), State(), State(), State()]

    ego = State()
    code_any = "any((any(other.y < 100 for other in others) and other.x -ego.x > 5 and other.type==Vehicle) for other in others)"
    # code_any = "all((other.x -ego.x > 5 and other.type==Vehicle) for other in others)"
    ast_any = ast.parse(code_any).body[0].value
    cont_var_dict = {
        "others.x":[1,2,4,5],
        "others.y":[3,4,5,6],
        "ego.x":5,
        "ego.y":2
    }
    disc_var_dict = {
        "others.type":['Vehicle','Sign','Vehicle','Obs'],
        "ego.type":'Ped'
    }
    len_dict = {
        "others":len(others)
    }
    res = unroll_any_all(ast_any, cont_var_dict, disc_var_dict, len_dict)
    print(astunparse.unparse(ast_any))
    print(astunparse.unparse(res))
