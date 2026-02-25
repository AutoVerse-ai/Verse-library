from z3 import *
from typing import List, Tuple, Dict, Callable, Any
import ast
import inspect
import textwrap

def z3num_to_float(x):
    """Convert Z3 numeric expressions to Python float, handling rationals and epsilon terms."""
    x = simplify(x)
    if isinstance(x, (IntNumRef, RatNumRef, AlgebraicNumRef)):
        s = x.as_string().rstrip('?')
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)
    s = str(x).replace("epsilon", "0")

    try: 
        return float(eval(s))
    except:
        raise ValueError(f"Cannot parse Z3 numeric expression: {x}")
    
def bounded_piecewise_z3(conditions_outputs, input_bounds):
    """
    Evaluate a piecewise function over bounded inputs using Z3.
    For each branch, finds the tightest input bounds satisfying that branch's condition.
    """
    results = []
    var_names = list(input_bounds.keys())
    z3_vars = {name: Real(name) for name in var_names}
    negated_conditions = []
    
    for _, (condition, output_func) in enumerate(conditions_outputs):
        if condition is None:
            branch_condition = And(negated_conditions) if negated_conditions else BoolVal(True)
        else:
            branch_condition = condition
            negated_conditions.append(Not(condition))
        
        constrained_bounds = {}
        for name in var_names:
            opt_min = Optimize()
            for bound_name, (min_val, max_val) in input_bounds.items():
                opt_min.add(z3_vars[bound_name] >= min_val)
                opt_min.add(z3_vars[bound_name] <= max_val)
            opt_min.add(branch_condition)
            obj = opt_min.minimize(z3_vars[name])

            min_val = None
            if opt_min.check() == sat:
                inf = opt_min.lower(obj)
                min_val = z3num_to_float(inf)

            opt_max = Optimize()
            for bound_name, (min_val_b, max_val_b) in input_bounds.items():
                opt_max.add(z3_vars[bound_name] >= min_val_b)
                opt_max.add(z3_vars[bound_name] <= max_val_b)
            opt_max.add(branch_condition)
            obj = opt_max.maximize(z3_vars[name])

            max_val = None
            if opt_max.check() == sat:
                sup = opt_max.upper(obj)
                max_val = z3num_to_float(sup)
            
            if min_val is not None and max_val is not None:
                constrained_bounds[name] = (min_val, max_val)
        
        if len(constrained_bounds) == len(var_names):
            results.append((constrained_bounds, output_func))
    
    return results

def parse_piecewise_function(func: Callable) -> List[Tuple[Any, Any]]:
    """
    Parse a Python piecewise function (if-elif-else) into conditions_outputs format.
    
    Returns:
        List of (condition_z3_expr, output_expr) Tuples
        where output_expr is either a constant or an AST expression string
    """
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Input must be a function definition")
    
    if_stmt = None
    for node in func_def.body:
        if isinstance(node, ast.If):
            if_stmt = node
            break
    
    if if_stmt is None:
        raise ValueError("Function must contain an if statement")
    
    arg_names = [arg.arg for arg in func_def.args.args]
    z3_vars = {name: Real(name) for name in arg_names}
    
    conditions_outputs = []
    
    current = if_stmt
    while current is not None:
        condition_z3 = ast_to_z3_condition(current.test, z3_vars)
        output_expr = extract_return_value(current.body)
        conditions_outputs.append((condition_z3, output_expr))
        
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
        elif len(current.orelse) > 0:
            output_expr = extract_return_value(current.orelse)
            conditions_outputs.append((None, output_expr))
            current = None
        else:
            current = None
    
    return conditions_outputs


def ast_to_z3_condition(node: ast.expr, z3_vars: Dict[str, Any]) -> Any:
    """
    Convert an AST condition expression to a Z3 boolean expression.
    """
    if isinstance(node, ast.Compare):
        left = ast_to_z3_expr(node.left, z3_vars)
        
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = ast_to_z3_expr(comparator, z3_vars)
            
            if isinstance(op, ast.Gt):
                comparison = left > right
            elif isinstance(op, ast.GtE):
                comparison = left >= right
            elif isinstance(op, ast.Lt):
                comparison = left < right
            elif isinstance(op, ast.LtE):
                comparison = left <= right
            elif isinstance(op, ast.Eq):
                comparison = left == right
            elif isinstance(op, ast.NotEq):
                comparison = left != right
            else:
                raise ValueError(f"Unsupported comparison operator: {op}")
            
            result = comparison if result is None else And(result, comparison)
        
        return result
    
    elif isinstance(node, ast.BoolOp):
        operands = [ast_to_z3_condition(val, z3_vars) for val in node.values]
        
        if isinstance(node.op, ast.And):
            return And(operands)
        elif isinstance(node.op, ast.Or):
            return Or(operands)
        else:
            raise ValueError(f"Unsupported boolean operator: {node.op}")
    
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return Not(ast_to_z3_condition(node.operand, z3_vars))
    
    else:
        raise ValueError(f"Unsupported condition type: {type(node)}")


def ast_to_z3_expr(node: ast.expr, z3_vars: Dict[str, Any]) -> Any:
    """
    Convert an AST expression to a Z3 expression.
    """
    if isinstance(node, ast.Name):
        if node.id in z3_vars:
            return z3_vars[node.id]
        else:
            raise ValueError(f"Unknown variable: {node.id}")
    
    elif isinstance(node, ast.Constant):
        return node.value
    
    elif isinstance(node, ast.Num):
        # NOTE: Python < 3.8 compatibility - ast.Num is deprecated in favor of ast.Constant
        return node.n
    
    elif isinstance(node, ast.UnaryOp):
        operand = ast_to_z3_expr(node.operand, z3_vars)
        
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operator: {node.op}")
    
    elif isinstance(node, ast.BinOp):
        left = ast_to_z3_expr(node.left, z3_vars)
        right = ast_to_z3_expr(node.right, z3_vars)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        else:
            raise ValueError(f"Unsupported binary operator: {node.op}")
    
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


def extract_return_value(body: List[ast.stmt]) -> Any:
    """
    Extract the return value from a function body.
    Returns either a constant value or unparsed AST expression string.
    """
    for node in body:
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Constant):
                return node.value.value
            elif isinstance(node.value, ast.Num):
                # NOTE: Python < 3.8 compatibility
                return node.value.n
            else:
                return ast.unparse(node.value)
    
    raise ValueError("No return statement found")


def create_output_func(output_expr: str, arg_names: List[str]) -> Callable:
    """
    Create a lambda function from an output expression string.
    Takes a dict of variable bounds and evaluates the expression.
    """
    def output_func(var_dict):
        expr = output_expr
        for name in arg_names:
            if name in var_dict:
                expr = expr.replace(name, str(var_dict[name]))
        
        return eval(expr)
    
    return output_func

if __name__ == "__main__":
    def vis_sensor_piecewise(psi, phi):
        """Example piecewise function for visibility sensor."""
        if psi >= 0 and psi <= phi:
            return -1
        elif psi < 0 and psi >= -phi:
            return 1
        elif psi > phi:
            return something(psi, 2)
        else:
            return 2

    conditions_outputs = parse_piecewise_function(vis_sensor_piecewise)

    input_bounds = {
        'psi': [-1.1, 2],
        'phi': [1, 1]
    }

    z3_vars = {'psi': Real('psi'), 'phi': Real('phi')}
    results = bounded_piecewise_z3(conditions_outputs, input_bounds)

    for constrained_bounds, output_func in results:
        print(f"Constrained bounds: {constrained_bounds}")
        print(f"Output function: {output_func}")

""" 
    NOTE: The most complex part requiring some attention is the Z3 optimization loop in
    bounded_piecewise_z3(). The function creates separate Optimize() instances for min/max
    per variable per branch, which can be slow for complex conditions or many variables.
    Consider caching or refactoring if performance becomes an issue. 
"""