from z3 import *
from typing import List, Tuple, Dict, Callable, Any
import ast
import inspect
import textwrap

def z3num_to_float(x):
    # Simplify expression (handles things like "-1 * epsilon")
    x = simplify(x)

    # Numeric types: Int, Rational, Algebraic
    if isinstance(x, (IntNumRef, RatNumRef, AlgebraicNumRef)):
        s = x.as_string().rstrip('?')
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)

    # Symbolic expressions involving epsilon → replace epsilon → 0
    s = str(x).replace("epsilon", "0")

    # The cleaned string may contain spaces/operators; extract the number:
    try:
        return float(eval(s))   # safe because s only contains 0, digits, +, -, *, /
    except:
        raise ValueError(f"Cannot parse Z3 numeric expression: {x}")
    
def bounded_piecewise_z3(conditions_outputs, input_bounds):
    """
    Evaluate a piecewise function over bounded inputs using Z3.
    """
    
    results = []
    var_names = list(input_bounds.keys())
    
    # Create Z3 variables
    z3_vars = {name: Real(name) for name in var_names}
    
    # Track which conditions have been checked (for else clause)
    negated_conditions = []
    
    for i, (condition, output_func) in enumerate(conditions_outputs):
        # Determine the condition for this branch
        if condition is None:
            # Else clause: negate all previous conditions
            if negated_conditions:
                branch_condition = And(negated_conditions)
            else:
                branch_condition = BoolVal(True)
        else:
            branch_condition = condition
            negated_conditions.append(Not(condition))
        
        # Branch is satisfiable, now find constrained bounds
        constrained_bounds = {}
        
        for name in var_names:
            # Find minimum value that satisfies both input bounds AND branch condition
            opt_min = Optimize()
            for bound_name, (min_val, max_val) in input_bounds.items():
                opt_min.add(z3_vars[bound_name] >= min_val)
                opt_min.add(z3_vars[bound_name] <= max_val)
            opt_min.add(branch_condition)
            # opt_min.minimize(z3_vars[name])
            obj = opt_min.minimize(z3_vars[name])

            min_val = None
            if opt_min.check() == sat:
            #     model = opt_min.model()
            #     min_val_expr = model[z3_vars[name]]
            #     min_val = float(min_val_expr.as_decimal(10))
                # Extract INFIMUM of the objective
                inf = opt_min.lower(obj)       # <-- This is the correct API for Optimize
                min_val = z3num_to_float(inf)

            # Find maximum value that satisfies both input bounds AND branch condition
            opt_max = Optimize()
            for bound_name, (min_val_b, max_val_b) in input_bounds.items():
                opt_max.add(z3_vars[bound_name] >= min_val_b)
                opt_max.add(z3_vars[bound_name] <= max_val_b)
            opt_max.add(branch_condition)
            # opt_max.maximize(z3_vars[name])
            obj = opt_max.maximize(z3_vars[name])
            

            max_val = None
            if opt_max.check() == sat:
                sup = opt_max.upper(obj)       # <-- This is the correct API for Optimize
                max_val = z3num_to_float(sup)
                # model = opt_max.model()
                # max_val_expr = model[z3_vars[name]]
                # max_val = float(max_val_expr.as_decimal(10))
            
            if min_val is not None and max_val is not None:
                constrained_bounds[name] = (min_val, max_val)
        
        # Add result if we got bounds for all variables
        if len(constrained_bounds) == len(var_names):
            results.append((constrained_bounds, output_func))
    
    return results

def parse_piecewise_function(func: Callable) -> List[Tuple[Any, Any]]:
    """
    Parse a Python piecewise function (if-elif-else) into conditions_outputs format.
    
    Returns:
        List of (condition_z3_expr, output_expr) tuples
        where output_expr is either a constant or an AST expression string
    """
    
    # Get the source code and parse it
    source = inspect.getsource(func)
    source = textwrap.dedent(source)  
    tree = ast.parse(source)
    
    # Find the function definition
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Input must be a function definition")
    
    # Extract the if-elif-else chain from the function body
    if_stmt = None
    for node in func_def.body:
        if isinstance(node, ast.If):
            if_stmt = node
            break
    
    if if_stmt is None:
        raise ValueError("Function must contain an if statement")
    
    # Get function arguments for variable mapping
    arg_names = [arg.arg for arg in func_def.args.args]
    z3_vars = {name: Real(name) for name in arg_names}
    
    conditions_outputs = []
    
    # Process the if-elif-else chain
    current = if_stmt
    while current is not None:
        # Parse the condition
        condition_z3 = ast_to_z3_condition(current.test, z3_vars)
        
        # Parse the output (keep as expression, don't wrap in function)
        output_expr = extract_return_value(current.body)
        
        conditions_outputs.append((condition_z3, output_expr))
        
        # Move to elif/else
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]  # elif
        elif len(current.orelse) > 0:
            # else clause
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
    
    elif isinstance(node, ast.Num):  # Python < 3.8 -- may be deprecated
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
    """
    for node in body:
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Constant):
                return node.value.value
            elif isinstance(node.value, ast.Num):  # Python < 3.8
                return node.value.n
            else:
                # Return the AST node for complex expressions
                return ast.unparse(node.value)
    
    raise ValueError("No return statement found")


def create_output_func(output_expr: str, arg_names: List[str]) -> Callable:
    """
    Create a lambda function from an output expression string.
    """
    # Create a function that takes a dict of variable bounds
    def output_func(var_dict):
        # Replace variable names with their values from var_dict
        expr = output_expr
        for name in arg_names:
            if name in var_dict:
                expr = expr.replace(name, str(var_dict[name]))
        # Evaluate the expression
        return eval(expr)
    
    return output_func

if __name__ == "__main__":

    # vis example:
    def vis_sensor_piecewise(psi, phi):
        if psi >= 0 and psi <= phi:
            return -1
        elif psi < 0 and psi >= -phi:
            return 1
        elif psi > phi:
            # return -2 
            return something(psi, 2) # parser should be able to parse out arbitrary functions
        else:
            return 2

    # Parsing
    conditions_outputs = parse_piecewise_function(vis_sensor_piecewise)

    # print(conditions_outputs)
    # Checking
    input_bounds = {
        'psi': [-1.1, 2],
        'phi': [1, 1]
    }

    z3_vars = {'psi': Real('psi'), 'phi': Real('phi')}
    results = bounded_piecewise_z3(conditions_outputs, input_bounds)

    for constrained_bounds, output_func in results:
        print(f"Constrained bounds: {constrained_bounds}")
        print(f"Output function: {output_func}")