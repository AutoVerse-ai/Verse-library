from typing import Any
import pickle
import ast

from z3 import *

from verse.map import LaneMap, AbstractLane
from verse.analysis.utils import *
from verse.agents.base_agent import BaseAgent
from verse.parser import Reduction, ReductionType, unparse

class LogicTreeNode:
    def __init__(self, data, child = [], val = None, mode_guard = None):
        self.data = data 
        self.child = child
        self.val = val
        self.mode_guard = mode_guard

class NodeSubstituter(ast.NodeTransformer):
    def __init__(self, old_node, new_node):
        super().__init__()
        self.old_node = old_node 
        self.new_node = new_node

    def visit_Reduction(self, node: Reduction) -> Any:
        if node == self.old_node:
            self.generic_visit(node)
            return self.new_node 
        else:
            self.generic_visit(node)
            return node

class ValueSubstituter(ast.NodeTransformer):
    def __init__(self, val:str, node):
        super().__init__()
        self.val = val
        self.node = node
    
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Substitute attribute node in the ast
        if isinstance(self.node, Reduction):
            return node
        if node == self.node:
            return ast.Name(
                id = self.val, 
                ctx = ast.Load()
            )
        return node

    def visit_Name(self, node: ast.Attribute) -> Any:
        # Substitute name node in the ast
        if isinstance(self.node, Reduction):
            return node
        if node == self.node:
            return ast.Name(
                id = self.val,
                ctx = ast.Load
            )
        return node

    def visit_Reduction(self, node: Reduction) -> Any:
        if node == self.node:
            if len(self.val) == 1:
                self.generic_visit(node)
                return self.val[0]
            elif node.op == ReductionType.Any:
                self.generic_visit(node)
                return ast.BoolOp(
                    op = ast.Or(),
                    values = self.val
                )
            elif node.op == ReductionType.All:
                self.generic_visit(node)
                return ast.BoolOp(
                    op = ast.And(),
                    values = self.val
                )
        self.generic_visit(node)
        return node


class GuardExpressionAst:
    def __init__(self, guard_list, guard_idx = 0):
        self.ast_list = copy.deepcopy(guard_list)
        self.cont_variables = {}
        self.varDict = {}
        self.guard_idx = guard_idx

    def _build_guard(self, guard_str, agent):
        """
        Build solver for current guard based on guard string

        Args:
            guard_str (str): the guard string.
            For example:"And(v>=40-0.1*u, v-40+0.1*u<=0)"

        Returns:
            A Z3 Solver obj that check for guard.
            A symbol index dic obj that indicates the index
            of variables that involved in the guard.
        """
        cur_solver = Solver()
        # This magic line here is because SymPy will evaluate == to be False
        # Therefore we are not be able to get free symbols from it
        # Thus we need to replace "==" to something else

        symbols_map = {v: k for k, v in self.cont_variables.items() if k in guard_str}

        for vars in reversed(self.cont_variables):
            guard_str = guard_str.replace(vars, self.cont_variables[vars])
        # XXX `locals` should override `globals` right?
        cur_solver.add(eval(guard_str, globals(), self.varDict))  # TODO use an object instead of `eval` a string
        return cur_solver, symbols_map

    def evaluate_guard_cont(self, agent, continuous_variable_dict, lane_map):
        res = False
        is_contained = False

        for cont_vars in continuous_variable_dict:
            underscored = cont_vars.replace('.','_')
            self.cont_variables[cont_vars] = underscored
            self.varDict[underscored] = Real(underscored)

        z3_string = self.generate_z3_expression() 
        if isinstance(z3_string, bool):
            return z3_string, z3_string 

        cur_solver, symbols = self._build_guard(z3_string, agent)
        cur_solver.push()
        for symbol in symbols:
            start, end = continuous_variable_dict[symbols[symbol]]
            cur_solver.add(self.varDict[symbol] >= start, self.varDict[symbol] <= end)
        if cur_solver.check() == sat:
            # The reachtube hits the guard
            cur_solver.pop()
            res = True
            
            tmp_solver = Solver()
            tmp_solver.add(Not(cur_solver.assertions()[0]))
            for symbol in symbols:
                start, end = continuous_variable_dict[symbols[symbol]]
                tmp_solver.add(self.varDict[symbol] >= start, self.varDict[symbol] <= end)
            if tmp_solver.check() == unsat:
                is_contained = True

        return res, is_contained

    def generate_z3_expression(self):
        """
        The return value of this function will be a bool/str

        If without evaluating the continuous variables the result is True, then
        the guard will automatically be satisfied and is_contained will be True

        If without evaluating the continuous variables the result is False, th-
        en the guard will automatically be unsatisfied

        If the result is a string, then continuous variables will be checked to
        see if the guard can be satisfied 
        """
        res = []
        for node in self.ast_list:
            tmp = self._generate_z3_expression_node(node)
            if isinstance(tmp, bool):
                if not tmp:
                    return False
                else:
                    continue
            res.append(tmp)
        if res == []:
            return True
        elif len(res) == 1:
            return res[0]
        res = "And("+",".join(res)+")"
        return res

    def _generate_z3_expression_node(self, node):
        """
        Perform a DFS over expression ast and generate the guard expression
        The return value of this function can be a bool/str

        If without evaluating the continuous variables the result is True, then
        the guard condition will automatically be satisfied
        
        If without evaluating the continuous variables the result is False, then
        the guard condition will not be satisfied

        If the result is a string, then continuous variables will be checked to
        see if the guard can be satisfied
        """
        if isinstance(node, ast.BoolOp):
            # Check the operator
            # For each value in the boolop, check results
            if isinstance(node.op, ast.And):
                z3_str = []
                for i,val in enumerate(node.values):
                    tmp = self._generate_z3_expression_node(val)
                    if isinstance(tmp, bool):
                        if tmp:
                            continue 
                        else:
                            return False
                    z3_str.append(tmp)
                if len(z3_str) == 1:
                    z3_str = z3_str[0]
                else:
                    z3_str = 'And('+','.join(z3_str)+')'
                return z3_str
            elif isinstance(node.op, ast.Or):
                z3_str = []
                for val in node.values:
                    tmp = self._generate_z3_expression_node(val)
                    if isinstance(tmp, bool):
                        if tmp:
                            return True
                        else:
                            continue
                    z3_str.append(tmp)
                if len(z3_str) == 1:
                    z3_str = z3_str[0]
                else:
                    z3_str = 'Or('+','.join(z3_str)+')'
                return z3_str
            # If string, construct string
            # If bool, check result and discard/evaluate result according to operator
            pass 
        elif isinstance(node, ast.Constant):
            # If is bool, return boolean result
            if isinstance(node.value, bool):
                return node.value
            # Else, return raw expression
            else:
                expr = unparse(node)
                expr = expr.strip('\n')
                return expr
        elif isinstance(node, ast.UnaryOp):
            # If is UnaryOp, 
            value = self._generate_z3_expression_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            elif isinstance(node.op, ast.Not):
                z3_str = 'Not('+value+')'
                return z3_str
            else:
                raise NotImplementedError(f"UnaryOp {node.op} is not supported")
        else:
            # For other cases, we can return the expression directly
            expr = unparse(node)
            expr = expr.strip('\n')
            return expr

    def evaluate_guard_hybrid(self, agent, discrete_variable_dict, continuous_variable_dict, lane_map:LaneMap):
        """
        Handle guard atomics that contains both continuous and hybrid variables
        Especially, we want to handle function calls that need both continuous and 
        discrete variables as input 
        We will perform interval arithmetic based on the function calls to the input and replace the function calls
        with temp constants with their values stored in the continuous variable dict
        By doing this, all calls that need both continuous and discrete variables as input will now become only continuous
        variables. We can then handle these using what we already have for the continous variables
        """
        res = True 
        for i, node in enumerate(self.ast_list):
            tmp, self.ast_list[i] = self._evaluate_guard_hybrid(node, agent, discrete_variable_dict, continuous_variable_dict, lane_map)
            res = res and tmp 
        return res

    def _evaluate_guard_hybrid(self, root, agent, disc_var_dict, cont_var_dict, lane_map:LaneMap):
        if isinstance(root, ast.Compare): 
            expr = unparse(root)
            left, root.left = self._evaluate_guard_hybrid(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            right, root.comparators[0] = self._evaluate_guard_hybrid(root.comparators[0], agent, disc_var_dict, cont_var_dict, lane_map)
            return True, root
        elif isinstance(root, ast.BoolOp):
            if isinstance(root.op, ast.And):
                res = True
                for i, val in enumerate(root.values):
                    tmp, root.values[i] = self._evaluate_guard_hybrid(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    res = res and tmp 
                    if not res:
                        break 
                return res, root 
            elif isinstance(root.op, ast.Or):
                res = False
                for val in root.values:
                    tmp,val = self._evaluate_guard_hybrid(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    res = res or tmp
                return res, root  
        elif isinstance(root, ast.BinOp):
            left, root.left = self._evaluate_guard_hybrid(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            right, root.right = self._evaluate_guard_hybrid(root.right, agent, disc_var_dict, cont_var_dict, lane_map)
            return True, root
        elif isinstance(root, ast.Call):
            if isinstance(root.func, ast.Attribute):
                func = root.func        
                if func.value.id == 'lane_map':
                    if func.attr == 'get_lateral_distance':
                        # Get function arguments
                        arg0_node = root.args[0]
                        arg1_node = root.args[1]
                        if isinstance(arg0_node, ast.Attribute):
                            arg0_var = arg0_node.value.id + '.' + arg0_node.attr
                        elif isinstance(arg0_node, ast.Name):
                            arg0_var = arg0_node.id
                        else:
                            raise ValueError(f"Node type {type(arg0_node)} is not supported")
                        vehicle_lane = disc_var_dict[arg0_var]
                        assert isinstance(arg1_node, ast.List)
                        arg1_lower = []
                        arg1_upper = []
                        for elt in arg1_node.elts:
                            if isinstance(elt, ast.Attribute):
                                var = elt.value.id + '.' + elt.attr
                            elif isinstance(elt, ast.Name):
                                var = elt.id
                            else:
                                raise ValueError(f"Node type {type(elt)} is not supported")
                            arg1_lower.append(cont_var_dict[var][0])
                            arg1_upper.append(cont_var_dict[var][1])   
                        vehicle_pos = (arg1_lower, arg1_upper)

                        # Get corresponding lane segments with respect to the set of vehicle pos
                        lane_seg1 = lane_map.get_lane_segment(vehicle_lane, arg1_lower)
                        lane_seg2 = lane_map.get_lane_segment(vehicle_lane, arg1_upper)

                        # Compute the set of possible lateral values with respect to all possible segments
                        lateral_set1 = self._handle_lateral_set(lane_seg1, np.array(vehicle_pos))
                        lateral_set2 = self._handle_lateral_set(lane_seg2, np.array(vehicle_pos))

                        # Use the union of two sets as the set of possible lateral positions
                        lateral_set = [min(lateral_set1[0], lateral_set2[0]), max(lateral_set1[1], lateral_set2[1])]
                        
                        # Construct the tmp variable
                        tmp_var_name = f'tmp_variable{len(cont_var_dict)+1}'
                        # Add the tmp variable to the cont var dict
                        cont_var_dict[tmp_var_name] = lateral_set
                        # Replace the corresponding function call in ast
                        root = ast.parse(tmp_var_name).body[0].value
                        return True, root
                    elif func.attr == 'get_longitudinal_position':
                        # Get function arguments
                        arg0_node = root.args[0]
                        arg1_node = root.args[1]
                        # assert isinstance(arg0_node, ast.Attribute)
                        if isinstance(arg0_node, ast.Attribute):
                            arg0_var = arg0_node.value.id + '.' + arg0_node.attr
                        elif isinstance(arg0_node, ast.Name):
                            arg0_var = arg0_node.id
                        else:
                            raise ValueError(f"Node type {type(arg0_node)} is not supported")
                        vehicle_lane = disc_var_dict[arg0_var]
                        assert isinstance(arg1_node, ast.List)
                        arg1_lower = []
                        arg1_upper = []
                        for elt in arg1_node.elts:
                            if isinstance(elt, ast.Attribute):
                                var = elt.value.id + '.' + elt.attr
                            elif isinstance(elt, ast.Name):
                                var = elt.id
                            else:
                                raise ValueError(f"Node type {type(elt)} is not supported")
                            arg1_lower.append(cont_var_dict[var][0])
                            arg1_upper.append(cont_var_dict[var][1])   
                        vehicle_pos = (arg1_lower, arg1_upper)

                        # Get corresponding lane segments with respect to the set of vehicle pos
                        lane_seg1 = lane_map.get_lane_segment(vehicle_lane, arg1_lower)
                        lane_seg2 = lane_map.get_lane_segment(vehicle_lane, arg1_upper)

                        # Compute the set of possible longitudinal values with respect to all possible segments
                        longitudinal_set1 = self._handle_longitudinal_set(lane_seg1, np.array(vehicle_pos))
                        longitudinal_set2 = self._handle_longitudinal_set(lane_seg2, np.array(vehicle_pos))

                        # Use the union of two sets as the set of possible longitudinal positions
                        longitudinal_set = [min(longitudinal_set1[0], longitudinal_set2[0]), max(longitudinal_set1[1], longitudinal_set2[1])]
                        
                        # Construct the tmp variable
                        tmp_var_name = f'tmp_variable{len(cont_var_dict)+1}'
                        # Add the tmp variable to the cont var dict
                        cont_var_dict[tmp_var_name] = longitudinal_set
                        # Replace the corresponding function call in ast
                        root = ast.parse(tmp_var_name).body[0].value
                        return True, root
                    else:
                        raise ValueError(f'Node type {func} from {unparse(func)} is not supported')
                else:
                    raise ValueError(f'Node type {func} from {unparse(func)} is not supported')
            else:
                raise ValueError(f'Node type {root.func} from {unparse(root.func)} is not supported')   
        elif isinstance(root, ast.Attribute):
            return True, root 
        elif isinstance(root, ast.Constant):
            return root.value, root 
        elif isinstance(root, ast.Name):
            return True, root
        elif isinstance(root, ast.UnaryOp):
            if isinstance(root.op, ast.USub):
                res, root.operand = self._evaluate_guard_hybrid(root.operand, agent, disc_var_dict, cont_var_dict, lane_map)
            elif isinstance(root.op, ast.Not):
                res, root.operand = self._evaluate_guard_hybrid(root.operand, agent, disc_var_dict, cont_var_dict, lane_map)
                if not res:
                    root.operand = ast.parse('False').body[0].value
                    return True, ast.parse('True').body[0].value
            else:
                raise ValueError(f'Node type {root} from {unparse(root)} is not supported')
            return True, root 
        else:
            raise ValueError(f'Node type {root} from {unparse(root)} is not supported')

    def _handle_longitudinal_set(self, lane_seg: AbstractLane, position: np.ndarray) -> List[float]:
        if lane_seg.type == "Straight":
            # Delta lower
            delta0 = position[0,:] - lane_seg.start
            # Delta upper
            delta1 = position[1,:] - lane_seg.start

            longitudinal_low = min(delta0[0]*lane_seg.direction[0], delta1[0]*lane_seg.direction[0]) + \
                min(delta0[1]*lane_seg.direction[1], delta1[1]*lane_seg.direction[1])
            longitudinal_high = max(delta0[0]*lane_seg.direction[0], delta1[0]*lane_seg.direction[0]) + \
                max(delta0[1]*lane_seg.direction[1], delta1[1]*lane_seg.direction[1])
            longitudinal_low += lane_seg.longitudinal_start
            longitudinal_high += lane_seg.longitudinal_start

            assert longitudinal_high >= longitudinal_low
            return longitudinal_low, longitudinal_high            
        elif lane_seg.type == "Circular":
            # Delta lower
            delta0 = position[0,:] - lane_seg.center
            # Delta upper
            delta1 = position[1,:] - lane_seg.center

            phi0 = np.min([
                np.arctan2(delta0[1], delta0[0]),
                np.arctan2(delta0[1], delta1[0]),
                np.arctan2(delta1[1], delta0[0]),
                np.arctan2(delta1[1], delta1[0]),
            ])
            phi1 = np.max([
                np.arctan2(delta0[1], delta0[0]),
                np.arctan2(delta0[1], delta1[0]),
                np.arctan2(delta1[1], delta0[0]),
                np.arctan2(delta1[1], delta1[0]),
            ])

            phi0 = lane_seg.start_phase + wrap_to_pi(phi0 - lane_seg.start_phase)
            phi1 = lane_seg.start_phase + wrap_to_pi(phi1 - lane_seg.start_phase)
            longitudinal_low = min(
                lane_seg.direction * (phi0 - lane_seg.start_phase)*lane_seg.radius,
                lane_seg.direction * (phi1 - lane_seg.start_phase)*lane_seg.radius
            ) + lane_seg.longitudinal_start
            longitudinal_high = max(
                lane_seg.direction * (phi0 - lane_seg.start_phase)*lane_seg.radius,
                lane_seg.direction * (phi1 - lane_seg.start_phase)*lane_seg.radius
            ) + lane_seg.longitudinal_start

            assert longitudinal_high >= longitudinal_low
            return longitudinal_low, longitudinal_high
        else:
            raise ValueError(f'Lane segment with type {lane_seg.type} is not supported')

    def _handle_lateral_set(self, lane_seg: AbstractLane, position: np.ndarray) -> List[float]:
        if lane_seg.type == "Straight":
            # Delta lower
            delta0 = position[0,:] - lane_seg.start
            # Delta upper
            delta1 = position[1,:] - lane_seg.start

            lateral_low = min(delta0[0]*lane_seg.direction_lateral[0], delta1[0]*lane_seg.direction_lateral[0]) + \
                min(delta0[1]*lane_seg.direction_lateral[1], delta1[1]*lane_seg.direction_lateral[1])
            lateral_high = max(delta0[0]*lane_seg.direction_lateral[0], delta1[0]*lane_seg.direction_lateral[0]) + \
                max(delta0[1]*lane_seg.direction_lateral[1], delta1[1]*lane_seg.direction_lateral[1])
            assert lateral_high >= lateral_low
            return lateral_low, lateral_high
        elif lane_seg.type == "Circular":
            dx = np.max([position[0,0]-lane_seg.center[0],0,lane_seg.center[0]-position[1,0]])
            dy = np.max([position[0,1]-lane_seg.center[1],0,lane_seg.center[1]-position[1,1]])
            r_low = np.linalg.norm([dx, dy])

            dx = np.max([np.abs(position[0,0]-lane_seg.center[0]),np.abs(position[1,0]-lane_seg.center[0])])
            dy = np.max([np.abs(position[0,1]-lane_seg.center[1]),np.abs(position[1,1]-lane_seg.center[1])])
            r_high = np.linalg.norm([dx, dy])
            lateral_low = min(lane_seg.direction*(lane_seg.radius - r_high),lane_seg.direction*(lane_seg.radius - r_low))
            lateral_high = max(lane_seg.direction*(lane_seg.radius - r_high),lane_seg.direction*(lane_seg.radius - r_low))
            # print(lateral_low, lateral_high)
            assert lateral_high >= lateral_low
            return lateral_low, lateral_high
        else:
            raise ValueError(f'Lane segment with type {lane_seg.type} is not supported')

    def evaluate_guard_disc(self, agent, discrete_variable_dict, continuous_variable_dict, lane_map):
        """
        Evaluate guard that involves only discrete variables. 
        """
        res = True
        for i, node in enumerate(self.ast_list):
            tmp, self.ast_list[i] = self._evaluate_guard_disc(node, agent, discrete_variable_dict, continuous_variable_dict, lane_map)
            res = res and tmp 
            if not res:
                break
        return res
            
    def _evaluate_guard_disc(self, root, agent:BaseAgent, disc_var_dict, cont_var_dict, lane_map):
        """
        Recursively called function to evaluate guard with only discrete variables
        The function will evaluate all guards with discrete variables and replace the nodes with discrete guards by
        boolean constants
        
        :params:
        :return: The return value will be a tuple. The first element in the tuple will either be a boolean value or a the evaluated value of of an expression involving guard
        The second element in the tuple will be the updated ast node 
        """
        if isinstance(root, ast.Compare):
            expr = unparse(root)
            left, root.left = self._evaluate_guard_disc(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            right, root.comparators[0] = self._evaluate_guard_disc(root.comparators[0], agent, disc_var_dict, cont_var_dict, lane_map)
            if isinstance(left, bool) or isinstance(right, bool):
                return True, root
            if isinstance(root.ops[0], ast.GtE):
                res = left>=right
            elif isinstance(root.ops[0], ast.Gt):
                res = left>right 
            elif isinstance(root.ops[0], ast.Lt):
                res = left<right
            elif isinstance(root.ops[0], ast.LtE):
                res = left<=right
            elif isinstance(root.ops[0], ast.Eq):
                res = left == right 
            elif isinstance(root.ops[0], ast.NotEq):
                res = left != right 
            else:
                raise ValueError(f'Node type {root} from {unparse(root)} is not supported')
            if res:
                root = ast.parse('True').body[0].value
            else:
                root = ast.parse('False').body[0].value    
            return res, root
        elif isinstance(root, ast.BoolOp):
            if isinstance(root.op, ast.And):
                res = True
                for i,val in enumerate(root.values):
                    tmp,root.values[i] = self._evaluate_guard_disc(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    res = res and tmp
                    if not res:
                        root = ast.Constant(value=False, kind=None)
                        break
                return res, root
            elif isinstance(root.op, ast.Or):
                res = False
                for i, val in enumerate(root.values):
                    tmp,root.values[i] = self._evaluate_guard_disc(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    res = res or tmp
                return res, root     
        elif isinstance(root, ast.BinOp):
            # Check left and right in the binop and replace all attributes involving discrete variables
            left, root.left = self._evaluate_guard_disc(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            right, root.right = self._evaluate_guard_disc(root.right, agent, disc_var_dict, cont_var_dict, lane_map)
            return True, root
        elif isinstance(root, ast.Call):
            expr = unparse(root)
            # Check if the root is a function
            if any([var in expr for var in disc_var_dict]) and all([var not in expr for var in cont_var_dict]):
                # tmp = re.split('\(|\)',expr)
                # while "" in tmp:
                #     tmp.remove("")
                # for arg in tmp[1:]:
                #     if arg in disc_var_dict:
                #         expr = expr.replace(arg,f'"{disc_var_dict[arg]}"')
                # res = eval(expr)
                for arg in disc_var_dict:
                    expr = expr.replace(arg, f'"{disc_var_dict[arg]}"')
                res = eval(expr)
                if isinstance(res, bool):
                    if res:
                        root = ast.parse('True').body[0].value
                    else:
                        root = ast.parse('False').body[0].value    
                else:
                    # # TODO-PARSER: Handle This
                    # for mode_name in agent.controller.mode_defs:
                    #     # TODO-PARSER: Handle This
                    #     if res in agent.controller.mode_defs[mode_name].modes:
                    #         res = mode_name+'.'+res
                    #         break
                    root = ast.parse(str(res)).body[0].value
                return res, root
            else:
                return True, root
        elif isinstance(root, ast.Attribute):
            expr = unparse(root)
            expr = expr.strip('\n')
            if expr in disc_var_dict:
                val = disc_var_dict[expr]
                # TODO-PARSER: Handle This
                # for mode_name in agent.controller.mode_defs:
                #     # TODO-PARSER: Handle This
                #     if val in agent.controller.mode_defs[mode_name].modes:
                #         val = mode_name+'.'+val
                #         break
                return val, root
            # TODO-PARSER: Handle This
            elif root.value.id in agent.controller.mode_defs:
                return expr, root
            else:
                return True, root
        elif isinstance(root, ast.Constant):
            return root.value, root
        elif isinstance(root, ast.UnaryOp):
            if isinstance(root.op, ast.USub):
                res, root.operand = self._evaluate_guard_disc(root.operand, agent, disc_var_dict, cont_var_dict, lane_map)
            elif isinstance(root.op, ast.Not):
                res, root.operand = self._evaluate_guard_disc(root.operand, agent, disc_var_dict, cont_var_dict, lane_map)
                if not res:
                    root.operand = ast.parse('False').body[0].value
                    return True, ast.parse('True').body[0].value
            else:
                raise ValueError(f'Node type {root} from {unparse(root)} is not supported')
            return True, root
        elif isinstance(root, ast.Name):
            expr = root.id
            if expr in disc_var_dict:
                val = disc_var_dict[expr]
                # # TODO-PARSER: Handle This
                # for mode_name in agent.controller.mode_defs:
                #     # TODO-PARSER: Handle This
                #     if val in agent.controller.mode_defs[mode_name].modes:
                #         val = mode_name + '.' + val 
                #         break 
                return val, root
            else:
                return True, root 
        else:
            raise ValueError(f'Node type {root} from {unparse(root)} is not supported')

    def evaluate_guard(self, agent, continuous_variable_dict, discrete_variable_dict, lane_map):
        res = True
        for i, node in enumerate(self.ast_list):
            tmp = self._evaluate_guard(node, agent, continuous_variable_dict, discrete_variable_dict, lane_map)
            res = tmp and res
            if not res:
                break
        return res

    def _evaluate_guard(self, root, agent, cnts_var_dict, disc_var_dict, lane_map):
        if isinstance(root, ast.Compare):
            left = self._evaluate_guard(root.left, agent, cnts_var_dict, disc_var_dict, lane_map)
            right = self._evaluate_guard(root.comparators[0], agent, cnts_var_dict, disc_var_dict, lane_map)
            if isinstance(root.ops[0], ast.GtE):
                return left>=right
            elif isinstance(root.ops[0], ast.Gt):
                return left>right 
            elif isinstance(root.ops[0], ast.Lt):
                return left<right
            elif isinstance(root.ops[0], ast.LtE):
                return left<=right
            elif isinstance(root.ops[0], ast.Eq):
                return left == right 
            elif isinstance(root.ops[0], ast.NotEq):
                return left != right 
            else:
                raise ValueError(f'Node type {root} from {unparse(root)} is not supported')

        elif isinstance(root, ast.BoolOp):
            if isinstance(root.op, ast.And):
                res = True
                for val in root.values:
                    tmp = self._evaluate_guard(val, agent, cnts_var_dict, disc_var_dict, lane_map)
                    res = res and tmp
                    if not res:
                        break
                return res
            elif isinstance(root.op, ast.Or):
                res = False
                for val in root.values:
                    tmp = self._evaluate_guard(val, agent, cnts_var_dict, disc_var_dict, lane_map)
                    res = res or tmp
                    if res:
                        break
                return res
        elif isinstance(root, ast.BinOp):
            left = self._evaluate_guard(root.left, agent, cnts_var_dict, disc_var_dict, lane_map)
            right = self._evaluate_guard(root.right, agent, cnts_var_dict, disc_var_dict, lane_map)
            if isinstance(root.op, ast.Sub):
                return left - right
            elif isinstance(root.op, ast.Add):
                return left + right
            else:
                raise ValueError(f'Node type {root} from {unparse(root)} is not supported')
        elif isinstance(root, ast.Call):
            expr = unparse(root)
            # Check if the root is a function
            if isinstance(root.func, ast.Attribute) and "map" in root.func.value.id:
            # if 'map' in expr:
                # tmp = re.split('\(|\)',expr)
                # while "" in tmp:
                #     tmp.remove("")
                # for arg in tmp[1:]:
                #     if arg in disc_var_dict:
                #         expr = expr.replace(arg,f'"{disc_var_dict[arg]}"')
                # res = eval(expr)
                for arg in disc_var_dict:
                    expr = expr.replace(arg, f'"{disc_var_dict[arg]}"')
                for arg in cnts_var_dict:
                    expr = expr.replace(arg, str(cnts_var_dict[arg]))    
                res = eval(expr)
                # TODO-PARSER: Handle This
                for mode_name in agent.controller.mode_defs:
                    # TODO-PARSER: Handle This
                    if res in agent.controller.mode_defs[mode_name].modes:
                        res = mode_name+'.'+res
                        break
                return res
        elif isinstance(root, ast.Attribute):
            expr = unparse(root)
            expr = expr.strip('\n')
            if expr in disc_var_dict:
                val = disc_var_dict[expr]
                # TODO-PARSER: Handle This
                for mode_name in agent.controller.mode_defs:
                    # TODO-PARSER: Handle This
                    if val in agent.controller.mode_defs[mode_name].modes:
                        val = mode_name+'.'+val
                        break
                return val
            elif expr in cnts_var_dict:
                val = cnts_var_dict[expr]
                return val

            # TODO-PARSER: Handle This
            elif root.value.id in agent.controller.mode_defs:
                return expr
        elif isinstance(root, ast.Constant):
            return root.value
        elif isinstance(root, ast.UnaryOp):
            val = self._evaluate_guard(root.operand, agent, cnts_var_dict, disc_var_dict, lane_map)
            if isinstance(root.op, ast.USub):
                return -val
            if isinstance(root.op, ast.Not):
                return not val
            else:
                raise ValueError(f'Node type {root} from {unparse(root)} is not supported')
        elif isinstance(root, ast.Name):
            variable = root.id 
            if variable in cnts_var_dict:
                val = cnts_var_dict[variable]
                return val 
            elif variable in disc_var_dict:
                val = disc_var_dict[variable]
                # TODO-PARSER: Handle This
                for mode_name in agent.controller.mode_defs:
                    # TODO-PARSER: Handle This
                    if val in agent.controller.mode_defs[mode_name].modes:
                        val = mode_name+'.'+val
                        break
                return val
            else:
                raise ValueError(f"{variable} doesn't exist in either continuous varibales or discrete variables") 
        else:
            print("agent", agent)
            raise ValueError(f'Node type {root} from {unparse(root)} is not supported')

    def parse_any_all(self, cont_var_dict: Dict[str, float], disc_var_dict: Dict[str, float], len_dict: Dict[str, int]) -> None: 
        for i in range(len(self.ast_list)):
            root = self.ast_list[i]
            j = 0
            while j < sum(1 for _ in ast.walk(root)):
                # TODO: Find a faster way to access nodes in the tree
                node = list(ast.walk(root))[j]
                if isinstance(node, ast.Call) and\
                    isinstance(node.func, ast.Name) and\
                    (node.func.id=='any' or node.func.id=='all'):
                    new_node = self.unroll_any_all(node, cont_var_dict, disc_var_dict, len_dict)
                    root = NodeSubstituter(node, new_node).visit(root)
                j += 1
            self.ast_list[i] = root 

    def unroll_any_all(
        self, node: ast.Call, 
        cont_var_dict: Dict[str, float], 
        disc_var_dict: Dict[str, float], 
        len_dict: Dict[str, float]
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
                parsed_elt = self._parse_elt(changed_elt, cont_var_dict, disc_var_dict, iter_name_list, targ_name_list, iter_pos_list)
                # Add the expanded elt into the list 
                expand_elt_ast_list.append(parsed_elt)
            # Create the new boolop (and/or) node based on the list of expanded elt
            return ValueSubstituter(expand_elt_ast_list, node).visit(node)
        else:
            return node

    def _parse_elt(self, root, cont_var_dict, disc_var_dict, iter_name_list, targ_name_list, iter_pos_list) -> Any:
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
                    # NOTE: At each time step, for each agent, the variable value mapping and their 
                    # sequence in the list is single. Therefore, for the same key, we will always rewrite 
                    # its content. 
                    variable_name = iter_name + '.' + node.attr
                    variable_val = None
                    if variable_name in cont_var_dict:
                        variable_val = cont_var_dict[variable_name][iter_pos]
                        cont_var_dict[tmp_variable_name] = variable_val
                    elif variable_name in disc_var_dict:
                        variable_val = disc_var_dict[variable_name][iter_pos]
                        disc_var_dict[tmp_variable_name] = variable_val

            elif isinstance(node, ast.Name):
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

    def parse_any_all_new(self, cont_var_dict: Dict[str, float], disc_var_dict: Dict[str, float], len_dict: Dict[str, int]) -> Dict[str, List[str]]: 
        cont_var_updater = {}
        for i in range(len(self.ast_list)):
            root = self.ast_list[i]
            nodes = ast.walk(root)
            for node in nodes:
                # TODO: Find a faster way to access nodes in the tree
                if isinstance(node, Reduction):
                    new_node = self.unroll_any_all_new(node, cont_var_dict, disc_var_dict, len_dict, cont_var_updater)
                    root = NodeSubstituter(node, new_node).visit(root)
            self.ast_list[i] = root 
        return cont_var_updater

    def unroll_any_all_new(
        self, node: ast.Call, 
        cont_var_dict: Dict[str, float], 
        disc_var_dict: Dict[str, float], 
        len_dict: Dict[str, float],
        cont_var_updater: Dict[str, List[str]],
    ) -> Tuple[ast.BoolOp, Dict[str, List[str]]]:
        # if isinstance(parse_arg, ast.GeneratorExp):
        iter_name = node.value.id
        iter_name_list = [node.value.id]
        targ_name = node.it
        targ_name_list = [node.it]
        # targ_var_list = []
        # for var in cont_var_dict:
        #     if var.startswith(iter_name + '.'):
        #         tmp = var.split('.')[1]
        #         targ_var_list.append(targ_name + '.' + tmp)
        # for var in disc_var_dict:
        #     if var.startswith(iter_name + '.'):
        #         tmp = var.split('.')[1]
        #         targ_var_list.append(targ_name + '.' + tmp)
        
        iter_len_list = [range(len_dict[node.value.id])]
        # Get all the iter, targets and the length of iter list 
        # for generator in parse_arg.generators:
        #     iter_name_list.append(generator.iter.id) # a_list
        #     targ_name_list.append(generator.target.id) # a
        #     iter_len_list.append(range(len_dict[generator.iter.id])) # len(a_list)

        elt = node.expr
        expand_elt_ast_list = []
        iter_len_list = list(itertools.product(*iter_len_list))
        # Loop through all possible combination of iter value
        for i in range(len(iter_len_list)):
            changed_elt = copy.deepcopy(elt)
            iter_pos_list = iter_len_list[i]
            # substitute temporary variable in each of the elt and add corresponding variables in the variable dicts
            parsed_elt = self._parse_elt_new(changed_elt, cont_var_dict, disc_var_dict, cont_var_updater, iter_name_list, targ_name_list, iter_pos_list)
            # Add the expanded elt into the list 
            expand_elt_ast_list.append(parsed_elt)
        # Create the new boolop (and/or) node based on the list of expanded elt
        return ValueSubstituter(expand_elt_ast_list, node).visit(node)

    def _parse_elt_new(self, root, cont_var_dict, disc_var_dict, cont_var_updater, iter_name_list, targ_var_list, iter_pos_list) -> Any:
        # Loop through all node in the elt ast 
        for node in ast.walk(root):
            # If the node is an attribute
            # Assume all the nodes are attributes
            if isinstance(node, ast.Attribute):
                node: ast.Attribute
                if node.value.id in targ_var_list:
                    # Find corresponding targ_name in the targ_var_list
                    targ_name = node.value.id
                    var_index = targ_var_list.index(targ_name)

                    # Find the corresponding iter_name in the iter_name_list 
                    iter_name = iter_name_list[var_index]

                    # Create the name for the tmp variable 
                    iter_pos = iter_pos_list[var_index]
                    tmp_variable_name = f"{iter_name}_{iter_pos}.{node.attr}"

                    # Replace variables in the etl by using tmp variables
                    root = ValueSubstituter(tmp_variable_name, node).visit(root)

                    # Find the value of the tmp variable in the cont/disc_var_dict
                    # Add the tmp variables into the cont/disc_var_dict
                    # NOTE: At each time step, for each agent, the variable value mapping and their 
                    # sequence in the list is single. Therefore, for the same key, we will always rewrite 
                    # its content. 
                    variable_name = iter_name + '.' + node.attr
                    variable_val = None
                    if variable_name in cont_var_dict:
                        # variable_val = cont_var_dict[variable_name][iter_pos]
                        # cont_var_dict[tmp_variable_name] = variable_val
                        if variable_name not in cont_var_updater:
                            cont_var_updater[variable_name] = [(tmp_variable_name, iter_pos)]
                        else:
                            if (tmp_variable_name, iter_pos) not in cont_var_updater[variable_name]:
                                cont_var_updater[variable_name].append((tmp_variable_name, iter_pos))
                    elif variable_name in disc_var_dict:
                        variable_val = disc_var_dict[variable_name][iter_pos]
                        disc_var_dict[tmp_variable_name] = variable_val
                        # if variable_name not in disc_var_updater:
                        #     disc_var_updater[variable_name] = [(tmp_variable_name, iter_pos)]
                        # else:
                        #     if (tmp_variable_name, iter_pos) not in disc_var_updater[variable_name]:
                        #         disc_var_updater[variable_name].append((tmp_variable_name, iter_pos))

            # elif isinstance(node, ast.Name):
            #     if node.id in targ_var_list:
            #         node:ast.Name
            #         # Find corresponding targ_name in the targ_var_list
            #         targ_name = node.id
            #         var_index = targ_var_list.index(targ_name)
            #         attr = node.id.split('.')[1]

            #         # Find the corresponding iter_name in the iter_name_list 
            #         iter_name = iter_name_list[var_index]

            #         # Create the name for the tmp variable 
            #         iter_pos = iter_pos_list[var_index]
            #         tmp_variable_name = f"{iter_name}_{iter_pos}.{attr}"

            #         # Replace variables in the etl by using tmp variables
            #         root = ValueSubstituter(tmp_variable_name, node).visit(root)

            #         # Find the value of the tmp variable in the cont/disc_var_dict
            #         # Add the tmp variables into the cont/disc_var_dict
            #         variable_name = iter_name + '.' + attr
            #         variable_val = None
            #         if variable_name in cont_var_dict:
            #             # variable_val = cont_var_dict[variable_name][iter_pos]
            #             # cont_var_dict[tmp_variable_name] = variable_val
            #             if variable_name not in cont_var_updater:
            #                 cont_var_updater[variable_name] = [(tmp_variable_name, iter_pos)]
            #             else:
            #                 if (tmp_variable_name, iter_pos) not in cont_var_updater[variable_name]:
            #                     cont_var_updater[variable_name].append((tmp_variable_name, iter_pos))
            #         elif variable_name in disc_var_dict:
            #             variable_val = disc_var_dict[variable_name][iter_pos]
            #             disc_var_dict[tmp_variable_name] = variable_val
                        # if variable_name not in disc_var_updater:
                        #     disc_var_updater[variable_name] = [(tmp_variable_name, iter_pos)]
                        # else:
                        #     if (tmp_variable_name, iter_pos) not in disc_var_updater[variable_name]:
                        #         disc_var_updater[variable_name].append((tmp_variable_name, iter_pos))
                # raise NotImplementedError("Let's just give up for now and worry about names later")
        # Return the modified node
        return root

if __name__ == "__main__":
    with open('tmp.pickle','rb') as f:
        guard_list = pickle.load(f)
    tmp = GuardExpressionAst(guard_list)
    # tmp.evaluate_guard()
    # tmp.construct_tree_from_str('(other_x-ego_x<20) and other_x-ego_x>10 and other_vehicle_lane==ego_vehicle_lane')
    print("stop")