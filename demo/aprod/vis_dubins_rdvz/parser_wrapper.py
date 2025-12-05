import ast
import inspect
import textwrap
from typing import Callable, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from sensor_parser import parsed_sensor_expr

def process_function_line_by_line(func: Callable, input_bounds: Dict[str, Tuple[float, float]]):
    """
    Process a function line-by-line, computing bounds at each step using parsed_sensor.
    
    Args:
        func: Function to analyze (must contain only assignments and return)
        input_bounds: Dict mapping variable names to (min, max) tuples
    
    Returns:
        Dict mapping variable names to their bounds at each step
    """
    
    # Parse the function AST
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    func_def = tree.body[0]
    
    # Track bounds of all variables
    current_bounds = dict(input_bounds)
    bounds_history = [dict(input_bounds)]
    
    # Process each statement
    for stmt in func_def.body:
        if isinstance(stmt, ast.Return):
            break
        
        elif isinstance(stmt, ast.Assign):
            # Single assignment: var_name = expression
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple assignments supported")
            
            var_name = stmt.targets[0].id
            rhs_expr = stmt.value
            
            # Convert RHS expression back to code string
            rhs_code = ast.unparse(rhs_expr)
            
            # Get bounds for this expression using parsed_sensor
            rhs_bounds = compute_expression_bounds_with_parsed_sensor(
                rhs_code,
                list(current_bounds.keys()),
                current_bounds
            )
            
            # Update variable bounds
            current_bounds[var_name] = rhs_bounds
            bounds_history.append(dict(current_bounds))
            
            print(f"After {var_name} = {rhs_code}")
            print(f"  {var_name} ∈ [{rhs_bounds[0]:.6f}, {rhs_bounds[1]:.6f}]")
    
    return current_bounds, bounds_history


def compute_expression_bounds_with_parsed_sensor(
    expr_code: str,
    arg_names: List[str],
    current_bounds: Dict[str, Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Use parsed_sensor to compute bounds for an expression.
    
    Args:
        expr_code: Expression as string, e.g., "x**2 + y**2"
        arg_names: List of argument names, e.g., ["x", "y"]
        current_bounds: Current variable bounds
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    
    # Create the temporary function source code -- add nonce for caching
    func_source = f"""
def temp_func({', '.join(arg_names)}):
    return {expr_code}
"""
    
    # Create module from string
    # module = TorchFuncModuleFromString(func_source, arg_names)
    
    # Convert bounds dict to list in the order of function arguments
    bounds_list = [current_bounds[name] for name in arg_names]
    
    # Use parsed_sensor to compute bounds
    try:
        lb, ub = parsed_sensor_expr(func_source, input_bounds=bounds_list)
        
        # Handle case where output is array
        if isinstance(lb, np.ndarray):
            lb = float(lb[0])
        if isinstance(ub, np.ndarray):
            ub = float(ub[0])
        
        return (lb, ub)
    
    except Exception as e:
        print(f"Error computing bounds: {e}")
        raise

# TODO: I'm not sure how big of a deal this is, but for usability, it'd be I could extract and create lists, e.g., a, b = c; c = a,b
# I think this is a little complex for now, however

# Usage example:
if __name__ == "__main__":
    def test_sensor(x, y):
        z = x**2 + y**2
        z = torch.sqrt(z) + torch.sum(torch.stack([x,y], dim=-1), dim=-1) # this is the correct way to do sums
        # z = torch.sqrt(z) + x + y
        return z
    
    input_bounds = {'x': (0, 1), 'y': (1, 2)}
    
    final_bounds, history = process_function_line_by_line(test_sensor, input_bounds)
    
    print("\nFinal bounds:")
    for var, bounds in final_bounds.items():
        print(f"  {var} ∈ [{bounds[0]:.6f}, {bounds[1]:.6f}]")