import enum
import re
from typing import List, Dict, Optional, Tuple, Any
import pickle
# from scene_verifier.automaton.hybrid_io_automaton import HybridIoAutomaton
# from pythonparser import Guard
import ast
import copy

from z3 import *
import astunparse
import numpy as np

from dryvr_plus_plus.scene_verifier.map.lane_map import LaneMap
from dryvr_plus_plus.scene_verifier.map.lane_segment import AbstractLane
from dryvr_plus_plus.scene_verifier.utils.utils import *
class LogicTreeNode:
    def __init__(self, data, child = [], val = None, mode_guard = None):
        self.data = data 
        self.child = child
        self.val = val
        self.mode_guard = mode_guard

class GuardExpressionAst:
    def __init__(self, guard_list):
        self.ast_list = []
        for guard in guard_list:
            self.ast_list.append(copy.deepcopy(guard.ast))
        self.cont_variables = {}
        self.varDict = {'t':Real('t')}

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

        symbols_map = {v: k for k, v in self.cont_variables.items()}

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
            if z3_string:
                return True, True 
            else:
                return False, False

        cur_solver, symbols = self._build_guard(z3_string, agent)
        cur_solver.push()
        for symbol in symbols:
            start, end = continuous_variable_dict[symbols[symbol]]
            cur_solver.add(self.varDict[symbol] >= start, self.varDict[symbol] <= end)
        if cur_solver.check() == sat:
            # The reachtube hits the guard
            cur_solver.pop()
            res = True
            
            # TODO: If the reachtube completely fall inside guard, break
            tmp_solver = Solver()
            tmp_solver.add(Not(cur_solver.assertions()[0]))
            for symbol in symbols:
                start, end = continuous_variable_dict[symbols[symbol]]
                tmp_solver.add(self.varDict[symbol] >= start, self.varDict[symbol] <= end)
            if tmp_solver.check() == unsat:
                print("Full intersect, break")
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
                expr = astunparse.unparse(node)
                expr = expr.strip('\n')
                return expr
        elif isinstance(node, ast.UnaryOp):
            # If is UnaryOp, 
            value = self._generate_z3_expression_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
        else:
            # For other cases, we can return the expression directly
            expr = astunparse.unparse(node)
            expr = expr.strip('\n')
            return expr

    def evaluate_guard_hybrid(self, agent, discrete_variable_dict, continuous_variable_dict, lane_map:LaneMap) -> Optional[bool]:
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
            sub_res = self._evaluate_guard_hybrid(node, agent, discrete_variable_dict, continuous_variable_dict, lane_map)
            if sub_res == None:
                return None
            tmp, self.ast_list[i] = sub_res
            res = res and tmp 
        return res

    def _evaluate_guard_hybrid(self, root, agent, disc_var_dict, cont_var_dict, lane_map:LaneMap) -> Optional[Tuple[bool, ast.expr]]:
        if isinstance(root, ast.Compare): 
            expr = astunparse.unparse(root)
            sub_res = self._evaluate_guard_hybrid(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            if sub_res == None:
                return None
            left, root.left = sub_res
            sub_res = self._evaluate_guard_hybrid(root.comparators[0], agent, disc_var_dict, cont_var_dict, lane_map)
            if sub_res == None:
                return None
            right, root.comparators[0] = sub_res
            return True, root
        elif isinstance(root, ast.BoolOp):
            if isinstance(root.op, ast.And):
                res = True
                for i, val in enumerate(root.values):
                    sub_res = self._evaluate_guard_hybrid(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    if sub_res == None:
                        return None
                    tmp, root.values[i] = sub_res
                    res = res and tmp 
                    if not res:
                        break 
                return res, root 
            elif isinstance(root.op, ast.Or):
                for val in root.values:
                    tmp,val = self._evaluate_guard_hybrid(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    res = res or tmp
                    if res:
                        break
                return res, root  
        elif isinstance(root, ast.BinOp):
            sub_res = self._evaluate_guard_hybrid(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            if sub_res == None:
                return None
            left, root.left = sub_res
            sub_res = self._evaluate_guard_hybrid(root.right, agent, disc_var_dict, cont_var_dict, lane_map)
            if sub_res == None:
                return None
            right, root.right = sub_res
            return True, root
        elif isinstance(root, ast.Call):
            if isinstance(root.func, ast.Attribute):
                func = root.func        
                if func.value.id == 'lane_map':
                    if func.attr == 'get_lateral_distance':
                        # Get function arguments
                        arg0_node = root.args[0]
                        arg1_node = root.args[1]
                        assert isinstance(arg0_node, ast.Attribute)
                        arg0_var = arg0_node.value.id + '.' + arg0_node.attr
                        vehicle_lane = disc_var_dict[arg0_var]
                        assert isinstance(arg1_node, ast.List)
                        arg1_lower = []
                        arg1_upper = []
                        for elt in arg1_node.elts:
                            if isinstance(elt, ast.Attribute):
                                var = elt.value.id + '.' + elt.attr
                                arg1_lower.append(cont_var_dict[var][0])
                                arg1_upper.append(cont_var_dict[var][1])   
                        vehicle_pos = (arg1_lower, arg1_upper)

                        # Get corresponding lane segments with respect to the set of vehicle pos
                        lane_seg1 = lane_map.get_lane_segment(vehicle_lane, arg1_lower)
                        lane_seg2 = lane_map.get_lane_segment(vehicle_lane, arg1_upper)

                        if None in [lane_seg1, lane_seg2]:
                            print(vehicle_pos)
                            print("\x1b[31mbshhhhhh\x1b[0m")
                            return None

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
                        assert isinstance(arg0_node, ast.Attribute)
                        arg0_var = arg0_node.value.id + '.' + arg0_node.attr
                        vehicle_lane = disc_var_dict[arg0_var]
                        assert isinstance(arg1_node, ast.List)
                        arg1_lower = []
                        arg1_upper = []
                        for elt in arg1_node.elts:
                            if isinstance(elt, ast.Attribute):
                                var = elt.value.id + '.' + elt.attr
                                arg1_lower.append(cont_var_dict[var][0])
                                arg1_upper.append(cont_var_dict[var][1])   
                        vehicle_pos = (arg1_lower, arg1_upper)

                        # Get corresponding lane segments with respect to the set of vehicle pos
                        lane_seg1 = lane_map.get_lane_segment(vehicle_lane, arg1_lower)
                        lane_seg2 = lane_map.get_lane_segment(vehicle_lane, arg1_upper)

                        if None in [lane_seg1, lane_seg2]:
                            print(vehicle_pos)
                            print("\x1b[31mbshhhhhh\x1b[0m")
                            return None

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
                        raise ValueError(f'Node type {func} from {astunparse.unparse(func)} is not supported')
                else:
                    raise ValueError(f'Node type {func} from {astunparse.unparse(func)} is not supported')
            else:
                raise ValueError(f'Node type {root.func} from {astunparse.unparse(root.func)} is not supported')   
        elif isinstance(root, ast.Attribute):
            return True, root 
        elif isinstance(root, ast.Constant):
            return root.value, root 
        elif isinstance(root, ast.UnaryOp):
            if isinstance(root.op, ast.USub):
                sub_res = self._evaluate_guard_hybrid(root.operand, agent, disc_var_dict, cont_var_dict, lane_map)
                if sub_res == None:
                    return None
                res, root.operand = sub_res
            else:
                raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')
            return True, root 
        else:
            raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')

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
        return res
            
    def _evaluate_guard_disc(self, root, agent, disc_var_dict, cont_var_dict, lane_map):
        """
        Recursively called function to evaluate guard with only discrete variables
        The function will evaluate all guards with discrete variables and replace the nodes with discrete guards by
        boolean constants
        
        :params:
        :return: The return value will be a tuple. The first element in the tuple will either be a boolean value or a the evaluated value of of an expression involving guard
        The second element in the tuple will be the updated ast node 
        """
        if isinstance(root, ast.Compare):
            expr = astunparse.unparse(root)
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
                raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')
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
                        break
                return res, root
            elif isinstance(root.op, ast.Or):
                res = False
                for val in root.values:
                    tmp,val = self._evaluate_guard_disc(val, agent, disc_var_dict, cont_var_dict, lane_map)
                    res = res or tmp
                    if res:
                        break
                return res, root     
        elif isinstance(root, ast.BinOp):
            # Check left and right in the binop and replace all attributes involving discrete variables
            left, root.left = self._evaluate_guard_disc(root.left, agent, disc_var_dict, cont_var_dict, lane_map)
            right, root.right = self._evaluate_guard_disc(root.right, agent, disc_var_dict, cont_var_dict, lane_map)
            return True, root
        elif isinstance(root, ast.Call):
            expr = astunparse.unparse(root)
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
                    root = ast.parse(str(res)).body[0].value
                return res, root
            else:
                return True, root
        elif isinstance(root, ast.Attribute):
            expr = astunparse.unparse(root)
            expr = expr.strip('\n')
            if expr in disc_var_dict:
                val = disc_var_dict[expr]
                for mode_name in agent.controller.modes:
                    if val in agent.controller.modes[mode_name]:
                        val = mode_name+'.'+val
                        break
                return val, root
            elif root.value.id in agent.controller.modes:
                return expr, root
            else:
                return True, root
        elif isinstance(root, ast.Constant):
            return root.value, root
        elif isinstance(root, ast.UnaryOp):
            if isinstance(root.op, ast.USub):
                res, root.operand = self._evaluate_guard_disc(root.operand, agent, disc_var_dict, cont_var_dict, lane_map)
            else:
                raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')
            return True, root
        else:
            raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')

    def evaluate_guard(self, agent, continuous_variable_dict, discrete_variable_dict, lane_map):
        res = True
        for node in self.ast_list:
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
                raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')

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
                raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')
        elif isinstance(root, ast.Call):
            expr = astunparse.unparse(root)
            # Check if the root is a function
            if 'map' in expr:
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
                return res
        elif isinstance(root, ast.Attribute):
            expr = astunparse.unparse(root)
            expr = expr.strip('\n')
            if expr in disc_var_dict:
                val = disc_var_dict[expr]
                for mode_name in agent.controller.modes:
                    if val in agent.controller.modes[mode_name]:
                        val = mode_name+'.'+val
                        break
                return val
            elif expr in cnts_var_dict:
                val = cnts_var_dict[expr]
                return val
            elif root.value.id in agent.controller.modes:
                return expr
        elif isinstance(root, ast.Constant):
            return root.value
        elif isinstance(root, ast.UnaryOp):
            val = self._evaluate_guard(root.operand, agent, cnts_var_dict, disc_var_dict, lane_map)
            if isinstance(root.op, ast.USub):
                return -val
            else:
                raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')
        else:
            raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')

if __name__ == "__main__":
    with open('tmp.pickle','rb') as f:
        guard_list = pickle.load(f)
    tmp = GuardExpressionAst(guard_list)
    # tmp.evaluate_guard()
    # tmp.construct_tree_from_str('(other_x-ego_x<20) and other_x-ego_x>10 and other_vehicle_lane==ego_vehicle_lane')
    print("stop")
