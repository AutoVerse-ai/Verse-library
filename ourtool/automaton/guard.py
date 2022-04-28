import enum
import re
from typing import List, Dict
import pickle
# from ourtool.automaton.hybrid_io_automaton import HybridIoAutomaton
# from pythonparser import Guard
import ast

from pkg_resources import compatible_platforms 
import astunparse

class LogicTreeNode:
    def __init__(self, data, child = [], val = None, mode_guard = None):
        self.data = data 
        self.child = child
        self.val = val
        self.mode_guard = mode_guard

'''
class GuardExpression:
    def __init__(self, root:LogicTreeNode=None, logic_str:str=None, guard_list=None):
        self._func_dict = {}

        self.logic_tree_root = root
        self.logic_string = logic_str

        if self.logic_tree_root is None and logic_str is not None:
            self.construct_tree_from_str(logic_str)
        elif guard_list is not None:
            self.construct_tree_from_list(guard_list)

    def construct_tree_from_list(self, guard_list:List[Guard]):
        # guard_list = ['('+elem.code+')' for elem in guard_list]   
        tmp = []
        func_count = 0
        for guard in guard_list:
            if guard.func is not None:
                func_identifier = f'func{func_count}'
                self._func_dict[func_identifier] = guard.code
                tmp.append(f'({func_identifier})')
            else:
                tmp.append('('+guard.code+')')
            
        guard_str = ' and '.join(tmp)
        self.construct_tree_from_str(guard_str)

    def logic_string_split(self, logic_string):
        # Input:
        #   logic_string: str, a python logic expression
        # Output:
        #   List[str], a list of string containing atomics, logic operator and brackets
        # The function take a python logic expression and split the expression into brackets, atomics and logic operators
        # logic_string = logic_string.replace(' ','')
        res = re.split('( and )',logic_string)

        tmp = []
        for sub_str in res:
            tmp += re.split('( or )',sub_str)
        res = tmp

        tmp = []
        for sub_str in res:
            tmp += re.split('(\()',sub_str)
        res = tmp

        tmp = []
        for sub_str in res:
            tmp += re.split('(\))',sub_str)
        res = tmp

        while("" in res) :
            res.remove("")
        while(" " in res):
            res.remove(" ")
        for i,sub_str in enumerate(res):
            res[i]= sub_str.strip(' ')

        # Handle spurious brackets in the splitted string
        # Get all the index of brackets pairs in the splitted string
        # Construct brackets tree
        # class BracketTreeNode:
        #     def __init__(self):
        #         self.left_idx = None 
        #         self.right_idx = None 
        #         self.child = []
        bracket_stack = []
        for i in range(len(res)):
            if res[i] == "(":
                bracket_stack.append(i)
            elif res[i] == ")":
                left_idx = bracket_stack.pop()
                sub_list = res[left_idx:i+1]
                # Check for each brackets pairs if there's any logic operators in between
                # If no, combine things in between and the brackets together, reconstruct the list
                if "and" not in sub_list and "or" not in sub_list:   
                    res[left_idx] = "".join(sub_list)
                    for j in range(left_idx+1,i+1):
                        res[j] = ""

        # For each pair of logic operator
        start_idx = 0
        end_idx = 0
        for i in range(len(res)):
            if res[i]!="(":
                start_idx = i
                break
        
        for i in range(len(res)):
            if res[i] == "and" or res[i] == "or":
                end_idx = i 
                sub_list = res[start_idx:end_idx]
                # Check if there's any dangling brackents in between. 
                # If no, combine things between logic operators
                if "(" not in sub_list and ")" not in sub_list:
                    res[start_idx] = "".join(sub_list)
                    for j in range(start_idx+1, end_idx):
                        res[j] = ""
                start_idx = end_idx + 1
        while("" in res) :
            res.remove("")

        # Put back functions
        for i in range(len(res)):
            for key in self._func_dict:
                if key in res[i]:
                    res[i] = res[i].replace(key, self._func_dict[key])
            # if res[i] in self._func_dict:
            #     res[i] = self._func_dict[res[i]]
        return res 

    def construct_tree_from_str(self, logic_string:str):
        # Convert an infix expression notation to an expression tree
        # https://www.geeksforgeeks.org/program-to-convert-infix-notation-to-expression-tree/

        self.logic_string = logic_string
        logic_string = "(" + logic_string + ")"
        s = self.logic_string_split(logic_string)

        stN = []
        stC = []
        p = {}
        p["and"] = 1
        p["or"] = 1
        p[")"] = 0

        for i in range(len(s)):
            if s[i] == "(":
                stC.append(s[i])
            
            elif s[i] not in p:
                t = LogicTreeNode(s[i])
                stN.append(t)
            
            elif(p[s[i]]>0):
                while (len(stC) != 0 and stC[-1] != '(' and p[stC[-1]] >= p[s[i]]):
                                    # Get and remove the top element
                    # from the character stack
                    t = LogicTreeNode(stC[-1])
                    stC.pop()
    
                    # Get and remove the top element
                    # from the node stack
                    t1 = stN[-1]
                    stN.pop()
    
                    # Get and remove the currently top
                    # element from the node stack
                    t2 = stN[-1]
                    stN.pop()
    
                    # Update the tree
                    t.child = [t1, t2]
    
                    # Push the node to the node stack
                    stN.append(t)
                stC.append(s[i])
            elif (s[i] == ')'):
                while (len(stC) != 0 and stC[-1] != '('):
                    # from the character stack
                    t = LogicTreeNode(stC[-1])
                    stC.pop()
    
                    # Get and remove the top element
                    # from the node stack
                    t1 = stN[-1]
                    stN.pop()
    
                    # Get and remove the currently top
                    # element from the node stack
                    t2 = stN[-1]
                    stN.pop()
    
                    # Update the tree
                    t.child = [t1, t2]
    
                    # Push the node to the node stack
                    stN.append(t)
                stC.pop()
        t = stN[-1]
        self.logic_tree_root = t

    def generate_guard_string_python(self):
        return self._generate_guard_string_python(self.logic_tree_root)

    def _generate_guard_string_python(self, root: LogicTreeNode)->str:
        if root.data!="and" and root.data!="or":
            return root.data
        else:
            data1 = self._generate_guard_string_python(root.child[0])
            data2 = self._generate_guard_string_python(root.child[1])
            return f"({data1} {root.data} {data2})"

    def generate_guard_string(self):
        return self._generate_guard_string(self.logic_tree_root)

    def _generate_guard_string(self, root: LogicTreeNode)->str:
        if root.data!="and" and root.data!="or":
            return root.data
        else:
            data1 = self._generate_guard_string(root.child[0])
            data2 = self._generate_guard_string(root.child[1])
            if root.data == "and":
                return f"And({data1},{data2})"
            elif root.data == "or":
                return f"Or({data1},{data2})"

    def evaluate_guard(self, agent, continuous_variable_dict, discrete_variable_dict, lane_map):
        res = self._evaluate_guard(self.logic_tree_root, agent, continuous_variable_dict, discrete_variable_dict, lane_map)
        return res

    def _evaluate_guard(self, root, agent, cnts_var_dict, disc_var_dict, lane_map):
        if root.child == []:
            expr = root.data
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
            # Elif check if the root contain any discrete data
            else:
                is_mode_guard = False
                for key in disc_var_dict:
                    if key in expr:
                        is_mode_guard = True
                        val = disc_var_dict[key]
                        for mode_name in agent.controller.modes:
                            if val in agent.controller.modes[mode_name]:
                                val = mode_name+'.'+val
                                break
                        expr = expr.replace(key, val)
                if is_mode_guard:
                    # Execute guard, assign type and  and return result
                    root.mode_guard = True
                    expr = expr.strip('(')
                    expr = expr.strip(')')
                    expr = expr.replace(' ','')
                    expr = expr.split('==')
                    res = expr[0] == expr[1]
                    # res = eval(expr)
                    root.val = res 
                    return res
                # Elif have cnts variable guard handle cnts variable guard
                else:
                    for key in cnts_var_dict:
                       expr = expr.replace(key, str(cnts_var_dict[key]))
                    res = eval(expr) 
                    return res
        # For the two children, call _execute_guard and collect result
        res1 = self._evaluate_guard(root.child[0],agent,cnts_var_dict, disc_var_dict, lane_map)
        res2 = self._evaluate_guard(root.child[1],agent,cnts_var_dict, disc_var_dict, lane_map)
        # Evaluate result for current node
        if root.data == "and":
            res = res1 and res2 
        elif root.data == "or":
            res = res1 or res2
        else:
            raise ValueError(f"Invalid root data {root.data}")
        return res       

    def execute_guard(self, discrete_variable_dict:Dict) -> bool:
        # This function will execute guard, and remove guard related to mode from the tree
        # We can do this recursively
        res = self._execute_guard(self.logic_tree_root, discrete_variable_dict)
        
        return res

    def _execute_guard(self, root:LogicTreeNode, discrete_variable_dict:Dict) -> bool:
        # If is tree leaf
        if root.child == []:
            # Check if the expression involves mode
            expr = root.data
            is_mode_guard = False
            for key in discrete_variable_dict:
                if key in expr:
                    is_mode_guard = True
                    expr = expr.replace(key, discrete_variable_dict[key])
            if is_mode_guard:
                # Execute guard, assign type and  and return result
                root.mode_guard = True
                expr = expr.strip('(')
                expr = expr.strip(')')
                expr = expr.replace(' ','')
                expr = expr.split('==')
                res = expr[0] == expr[1]
                # res = eval(expr)
                root.val = res 
                return res
            # Otherwise, return True
            else: 
                root.mode_guard = False 
                root.val = True 
                return True
        # For the two children, call _execute_guard and collect result
        res1 = self._execute_guard(root.child[0],discrete_variable_dict)
        res2 = self._execute_guard(root.child[1],discrete_variable_dict)
        # Evaluate result for current node
        if root.data == "and":
            res = res1 and res2 
        elif root.data == "or":
            res = res1 or res2
        else:
            raise ValueError(f"Invalid root data {root.data}")
        
        # If the result is False, return False
        if not res:
            return False 
        # Else if any child have false result, remove that child
        else: 
            if not res1 or root.child[0].mode_guard:
                root.data = root.child[1].data
                root.val = root.child[1].val 
                root.mode_guard = root.child[1].mode_guard
                root.child = root.child[1].child
            elif not res2 or root.child[1].mode_guard:
                root.data = root.child[0].data
                root.val = root.child[0].val 
                root.mode_guard = root.child[0].mode_guard
                root.child = root.child[0].child
            return True
'''

class GuardExpressionAst:
    def __init__(self, guard_list):
        self.ast_list = []
        for guard in guard_list:
            self.ast_list.append(guard.ast)

    def evaluate_guard_cont(self, agent, continuous_variable_dict, lane_map):
        res = True 
        is_contained = True
        # TODO 

        return res, is_contained

    # def _evaluate_guard_cont(self, root, agent, cont_var_dict, lane_map):
    #     return False

    def evaluate_guard_disc(self, agent, discrete_variable_dict, lane_map):
        """
        Evaluate guard that involves only discrete variables. 
        """
        res = True
        for i, node in enumerate(self.ast_list):
            tmp, self.ast_list[i] = self._evaluate_guard_disc(node, agent, discrete_variable_dict, lane_map)
            res = res and tmp 
        return res
            
    def _evaluate_guard_disc(self, root, agent, disc_var_dict, lane_map):
        if isinstance(root, ast.Compare):
            expr = astunparse.unparse(root)
            if any([var in expr for var in disc_var_dict]):
                left, root.left = self._evaluate_guard_disc(root.left, agent, disc_var_dict, lane_map)
                right, root.comparators[0] = self._evaluate_guard_disc(root.comparators[0], agent, disc_var_dict, lane_map)
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
            else:
                return True, root
        elif isinstance(root, ast.BoolOp):
            if isinstance(root.op, ast.And):
                res = True
                for i,val in enumerate(root.values):
                    tmp,root.values[i] = self._evaluate_guard_disc(val, agent, disc_var_dict, lane_map)
                    res = res and tmp
                    if not res:
                        break
                return res, root
            elif isinstance(root.op, ast.Or):
                res = False
                for val in root.values:
                    tmp,val = self._evaluate_guard_disc(val, agent, disc_var_dict, lane_map)
                    res = res or tmp
                    if res:
                        break
                return res, root     
        elif isinstance(root, ast.BinOp):
            return True, root
        elif isinstance(root, ast.Call):
            expr = astunparse.unparse(root)
            # Check if the root is a function
            if any([var in expr for var in disc_var_dict]):
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
        else:
            raise ValueError(f'Node type {root} from {astunparse.unparse(root)} is not supported')

if __name__ == "__main__":
    with open('tmp.pickle','rb') as f:
        guard_list = pickle.load(f)
    tmp = GuardExpressionAst(guard_list)
    # tmp.evaluate_guard()
    # tmp.construct_tree_from_str('(other_x-ego_x<20) and other_x-ego_x>10 and other_vehicle_lane==ego_vehicle_lane')
    print("stop")