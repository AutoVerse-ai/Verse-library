import ast
import inspect
import textwrap
from typing import Callable, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from sensor_parser import parsed_sensor_expr
from wrapper_consts import ALIASES, ANGULAR_FUNCTIONS, MAP_FUNCTIONS
from prox_error_all_bounds import angular_span_rect_parser

def process_function_line_by_line(func: Callable, input_bounds: Dict[str, Tuple[float, float]], piecewise_functions: List[Callable] = None):
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
            break # TODO: instead of immediately breaking, first extract the arguments being returned
        
        elif isinstance(stmt, ast.Assign):
            # Single assignment: var_name = expression
            # NOTE: could eventually allow for more complex assignments, but this should be expressive enough for now
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
                current_bounds,
                piecewise_functions
            )
            
            # Update variable bounds
            current_bounds[var_name] = rhs_bounds
            bounds_history.append(dict(current_bounds))
            
            print(f"After {var_name} = {rhs_code}")
            print(f"  {var_name} ∈ [{rhs_bounds[0]:.6f}, {rhs_bounds[1]:.6f}]")

        else:
            raise Exception(f'Line is of type {type(stmt)} instead of assign or return.')
    return current_bounds, bounds_history # TODO: should only return the variables actually called in the return output


def compute_expression_bounds_with_parsed_sensor(
    expr_code: str,
    arg_names: List[str],
    current_bounds: Dict[str, Tuple[float, float]],
    piecewise_functions: List[Callable]
) -> Tuple[float, float]:
    """
    Use parsed_sensor to compute bounds for an expression.
    
    Args:
        expr_code: Expression as string, e.g., "x**2 + y**2"
        arg_names: List of argument names, e.g., ["x", "y"]
        current_bounds: Current variable bounds
            
    Returns:
        (lower_bound, upper_bound) tuple

    NOTE: have not handled way to handle angular variables that could potentially have 2 bounds instead of 1
    NOTE cont: the ideal way to handle this is to compute all combinations of bounds, then if the function if the function is angular, simply apply the angular bound combine function to get minimal representation
    NOTE cont: and non-angular functions should just combine all outputs to a single bound     
    """
    
    for alias, func_name in ALIASES.items(): # NOTE: may want to optimize this in the future
        if alias in expr_code:
            # Replace the alias with the actual function name
            expr_code = expr_code.replace(alias, func_name)

    # TODO: fix way args are being added, right now, it's pretty naive and substrings of function calls could potentially overlap with variable names
    if any(func in expr_code for func in ANGULAR_FUNCTIONS):
        # Handle angular functions
        func_name = [func for func in ANGULAR_FUNCTIONS if func in expr_code][0]
        args = [arg for arg in current_bounds.keys() if arg in expr_code]
        bounds = handle_angular_function(func_name, args, current_bounds)
        return bounds
    
    elif piecewise_functions is not None and any(func.__name__ in expr_code for func in piecewise_functions):
        # Handle piecewise functions
        func = next(func for func in piecewise_functions if func.__name__ in expr_code)
        bounds = handle_piecewise_function(func, args, current_bounds)
        # TODO: Implement handling for piecewise functions
        return bounds
    # Check if the expression contains any map functions
    elif any(func in expr_code for func in MAP_FUNCTIONS):
        # Handle map functions
        func_name = [func for func in MAP_FUNCTIONS if func in expr_code][0]
        args = [arg for arg in current_bounds.keys() if arg in expr_code]
        bounds = handle_map_function(func_name, args, current_bounds)
        return bounds

    # Create the temporary function source code -- add nonce for caching to temp_func name
    # NOTE: using the raw arg_names isn't necessarily problematic but it does seem a bit inefficient
    func_source = f"""
def temp_func({', '.join(arg_names)}):
    return {expr_code}
"""
    
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
        raise Exception(f"Error computing CROWN bounds: {e}")

def handle_angular_function(func_name: str, args: List[str], current_bounds: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Handle angular functions.
    
    Args:
        func_name: Name of the angular function, e.g., "sin"
        args: List of argument names, e.g., ["x"]
        current_bounds: Current variable bounds
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    func = globals()[func_name]
    
    # Call the function with the current bounds
    bounds_list = [current_bounds[name] for name in args]
    bounds = func(*bounds_list)
    
    return bounds


def handle_map_function(func_name: str, args: List[str], current_bounds: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Handle map functions.
    
    Args:
        func_name: Name of the map function, e.g., "map_func1"
        args: List of argument names, e.g., ["x"]
        current_bounds: Current variable bounds as a dict of str argument names -> tuple bounds 
    
    Returns:
        (lower_bound, upper_bound) tuple
    """

    # TODO: Implement handling for map functions
    # TODO: add arguments for trackmode and map -- not sure if I want to do this for other functions yet
    return (0.0, 1.0)

def handle_piecewise_function(func: Callable, args: List[str], current_bounds: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Handle map functions.
    
    Args:
        func_name: Name of the map function, e.g., "map_func1"
        args: List of argument names, e.g., ["x"]
        current_bounds: Current variable bounds as a dict of str argument names -> tuple bounds 
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    # TODO: implement this, preferably using parse_piecewise_function
    # NOTE: may have to move functions from bounded_piecewise to this file since I need to run some form of compute_expression_bounded on the outputs of the piecewise functions 

    return (0,1) 


# Usage example:
if __name__ == "__main__":
    def test_sensor(x, y):
        z = x**2 + y**2
        z = torch.sqrt(z) + torch.sum(torch.stack([x,y], dim=-1), dim=-1) # this is the correct way to do sums
        # z = torch.sqrt(z) + x + y
        w = arctan2(x, y)
        return z # currently this does nothing, a return function just needs to exist
    
    # TODO: add a way to automatically create the input bounds given a list of bounds and the function header -- user shouldn't need to do this by hand
    input_bounds = {'x': (0, 1), 'y': (1, 2)}
    # input_bounds = {'x': (-1, -0.5), 'y': (-1, 1)} # this will error out since I don't have a way to handle lists of tuples yet
    

    final_bounds, history = process_function_line_by_line(test_sensor, input_bounds)
    
    print("\nFinal bounds:")
    for var, bounds in final_bounds.items():
        print(f"  {var} ∈ [{bounds[0]:.6f}, {bounds[1]:.6f}]")