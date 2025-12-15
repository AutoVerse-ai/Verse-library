import ast
import inspect
import textwrap
from typing import Callable, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from sensor_parser import parsed_sensor_expr
from wrapper_consts import ALIASES, ANGULAR_FUNCTIONS, MAP_FUNCTIONS
# ,FUNC_DICT
from prox_error_all_bounds import angular_span_rect_parser, angular_bounds_diff_correct, combine_angular_bounds
from bounded_map import get_heading_bounds_optimized, get_lateral_distance_bounds_optimized
import itertools
from bounded_piecewise import bounded_piecewise_z3, parse_piecewise_function
import pickle
import os
from pathlib import Path
import hashlib

# NOTE: same caching system as parsed_sensor -- use function name and input bounds to cache
def get_cache_key(input_bounds, function_name):
    """Generate a unique key for the input bounds and function name"""
    # Convert bounds to string and combine with function name
    bounds_str = str(input_bounds)
    combined_str = f"{bounds_str}_{function_name}"
    return hashlib.md5(combined_str.encode()).hexdigest()

def clear_parse_cache():
    """Clear all cached parse_function results"""
    cache_dir = Path(__file__).parent / "parse_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("parse_*.pkl"):
            f.unlink()

def parse_function(func: Callable, input_bounds: Dict[str, List[Tuple[float, float]]], piecewise_functions: List[Callable] = None, track_mode = None, track_map = None, cache: bool = True):
    """
    Process a function line-by-line, computing bounds at each step using parsed_sensor.
    
    Args:
        func: Function to analyze (must contain only assignments and return)
        input_bounds: Dict mapping variable names to (min, max) tuples
    
    Returns:
        Dict mapping variable names to their bounds at each step
    """
    
    # Create cache directory if it doesn't exist
    cache_dir = Path(__file__).parent / "parse_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key from input bounds and function name
    cache_key = get_cache_key(input_bounds, func.__name__)
    cache_file = cache_dir / f"parse_{cache_key}.pkl"
    
    # Check if cached result exists
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
        
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
                piecewise_functions,
                track_mode,
                track_map
            )
            
            # Update variable bounds
            current_bounds[var_name] = rhs_bounds
            bounds_history.append(dict(current_bounds))
            
            print(f"After {var_name} = {rhs_code}")
            for rhs_bound in rhs_bounds:
                print(f"  {var_name} ∈ [{rhs_bound[0]:.6f}, {rhs_bound[-1]:.6f}]")

        else:
            raise Exception(f'Line is of type {type(stmt)} instead of assign or return.')
    
    # NOTE: cache result if desired
    if cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((current_bounds, bounds_history), f)

    return current_bounds, bounds_history # TODO: should only return the variables actually called in the return output


def compute_expression_bounds_with_parsed_sensor(
    expr_code: str,
    arg_names: List[str],
    current_bounds: Dict[str, List[Tuple[float, float]]],
    piecewise_functions: List[Callable] = None,
    track_mode = None,
    track_map = None,
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
        the ideal way to handle this is to compute all combinations of bounds, then if the function if the function is angular, simply apply the angular bound combine function to get minimal representation
        and non-angular functions should just combine all outputs to a single bound     
    
    NOTE: should I let piecewise functions take mode as an input?

    FIXME: input args are not ordered properly -- should be ordered according to when they show up in the function
        this is an issue with ALL functions, although CROWN is likely already handled properly since we define the temp function args ourselves
    NOTE: I think I fixed this, for example, w = atan2(y,x) != m = atan2(x,y) given x = [-1, -0.5], y = [-1,0.5], but further testing needed
    """
    
    for alias, func_name in ALIASES.items(): # NOTE: may want to optimize this in the future
        if alias in expr_code:
            # Replace the alias with the actual function name
            expr_code = expr_code.replace(alias, func_name)

    # NOTE: extract args from function in the correct order
    tree = ast.parse(expr_code)
    func_def = tree.body[0].value

    # Extract the arguments from the func_code 
    # NOTE: this works in all cases except CROWN, where we expect general expressions, not function calls
    args = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call):
            args = [arg.id for arg in node.args if isinstance(arg, ast.Name) and arg.id in arg_names]
            break

    # TODO: fix way args are being added, right now, it's pretty naive and substrings of function calls could potentially overlap with variable names
    if any(func in expr_code for func in ANGULAR_FUNCTIONS):
        # Handle angular functions
        func_name = [func for func in ANGULAR_FUNCTIONS if func in expr_code][0]
        # args = [arg for arg in current_bounds.keys() if arg in expr_code]
        bounds = handle_angular_function(func_name, args, current_bounds)
        return bounds
    
    elif piecewise_functions is not None and any(func.__name__ in expr_code for func in piecewise_functions):
        # Handle piecewise functions
        func = next(func for func in piecewise_functions if func.__name__ in expr_code)
        # TODO: handle args correctly, currently just passing everything which isn't correct, shouldn't be too difficult to extract the args from the piecewise function itself
        # TODO: Implement handling for piecewise functions
        bounds = handle_piecewise_function(func, args, current_bounds)
        return bounds
    
    # Check if the expression contains any map functions
    elif any(func in expr_code for func in MAP_FUNCTIONS):
        # Handle map functions
        func_name = [func for func in MAP_FUNCTIONS if func in expr_code][0]
        # args = [arg for arg in current_bounds.keys() if arg in expr_code]
        bounds = handle_map_function(func_name, args, current_bounds, track_map, track_mode)
        return bounds

    # Create the temporary function source code -- add nonce for caching to temp_func name
    # NOTE: using the raw arg_names isn't necessarily problematic but it does seem a bit inefficient
    else:
        bounds = handle_crown_function(expr_code, arg_names, current_bounds)
        return bounds


def handle_angular_function(func_name: str, args: List[str], current_bounds: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
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
    # func = FUNC_DICT[func_name]
    
    # Call the function with the current bounds
    bounds_list = [current_bounds[name] for name in args]
    output_bounds = []
    
    # TODO(OPT): eventually this should be done in compute_expression_bounds once args are correctly extracted, do this once piecewise is handled
    # Iterate over all possible combinations of bounds -- *bounds_list forms a set of list of tuples, and tuples are sampled out of each list
    for bounds_combination in itertools.product(*bounds_list):
        # Call the function with the current bounds combination
        try:
            bounds = func(*bounds_combination)
            output_bounds.append(bounds)
        except Exception as e:
            raise Exception(f"Error computing angular bounds: {e}")
    
    # Combine the output bounds using combine_angular_bounds
    # TODO(OPT): pretty sure combine_angular_bounds function is way more complex than it needs to be -- why are positive and negative bounds being separately processed? they should be able to processed all at once
    combined_bounds = combine_angular_bounds(output_bounds)
    
    return combined_bounds


def handle_map_function(func_name: str, args: List[str], current_bounds: Dict[str, List[Tuple[float, float]]], track_mode, track_map) -> Tuple[float, float]:
    """
    Handle map functions.
    
    Args:
        func_name: Name of the map function, e.g., "map_func1"
        args: List of argument names, e.g., ["x"]
        current_bounds: Current variable bounds as a dict of str argument names -> tuple bounds 
    
    Returns:
        (lower_bound, upper_bound) tuple
    
    NOTE: expecting map_function to be called of form map_function(x,y,...,z) and assuming alias will transform it to form map_function_real(track_map, track_mode, x, y, ...,z)
    """

    # TODO: Implement handling for map functions
    # TODO: add arguments for trackmode and map -- not sure if I want to do this for other functions yet
    func = globals()[func_name]
    # func = FUNC_DICT[func_name]
    
    # Call the function with the current bounds
    bounds_list = [current_bounds[name] for name in args] # NOTE: guaranteed to be in right order if compute_expr_bounds is correct
    output_bounds = []

    for bounds_combination in itertools.product(*bounds_list):
        # Use parsed_sensor to compute bounds
        try:
            bounds = func(track_mode, track_map, *bounds_combination)
            output_bounds.append(bounds)
        except Exception as e:
            raise Exception(f"Error computing map function bounds: {e}")
    
    # Return the minimum and maximum bounds
    return [(min(np.array(output_bounds)[:,0]), max(np.array(output_bounds)[:,1]))] # NOTE: pretty sure this is correct, but could be wrong

def handle_piecewise_function(func: Callable, args: List[str], current_bounds: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
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
    bounds_list = [current_bounds[name] for name in args] 
    all_output_bounds = []
    parsed_func = parse_piecewise_function(func)

    for bounds_combination in itertools.product(*bounds_list):
        # Use parsed_sensor to compute bounds
        try:
            bounds_dict = dict(zip(args, bounds_combination)) # NOTE: due to how bounds_list is set up, creating the dict like this should keep the ordering 
            res = bounded_piecewise_z3(parsed_func, bounds_dict)
            for i in range(len(res)):
                output_function = str(res[i][1]) # NOTE: since res could be an int like 1
                constrained_bounds = {arg: [res[i][0][arg]] for arg in res[i][0]} # NOTE: this seems complex but is just taking the constrained bounds and converting from tuple -> list of a tuple, think res[i][0] is equivalent to using args directly
                output_bounds = compute_expression_bounds_with_parsed_sensor(
                    output_function,
                    args,
                    constrained_bounds,
                ) # NOTE: not allowing piecewise functions in piecewise functions, also not allowing map functions for time being
                all_output_bounds.extend(output_bounds) 

        except Exception as e:
            raise Exception(f"Error computing piecewise function bounds: {e}")

    # Compute the absolute minimum and maximum over the entire output_bounds list
    min_bound = min(min(output_bound) for output_bound in all_output_bounds)
    max_bound = max(max(output_bound) for output_bound in all_output_bounds)

    return [(min_bound, max_bound)]

def handle_crown_function(func_code: str, arg_names: List[str], current_bounds: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
    """
    Handle CROWN functions.
    
    Args:
        func_code: code of the RHS expression
        args: Complete list of argument names, e.g., ["x"]
        current_bounds: Current variable bounds as a dict of str argument names -> tuple bounds 
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    tree = ast.parse(func_code)
    func_def = tree.body[0].value

    # Extract the arguments from the func_code
    args = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and node.id in arg_names:
            args.append(node.id)

    # Remove duplicates and sort the arguments 
    # NOTE: why sort? 
    args = sorted(list(set(args)))
    # NOTE: haven't thoroughly vetted this code yet, worst case, just set args = arg_names if unconfident it will work

    func_source = f"""
def temp_func({', '.join(args)}):
    return {func_code}
"""
    
    # Convert bounds dict to list in the order of function arguments
    bounds_list = [current_bounds[name] for name in args]
    
    # Initialize lists to store the bounds
    lower_bounds = []
    upper_bounds = []
    
    # Iterate over all possible combinations of bounds
    for bounds_combination in itertools.product(*bounds_list):
        # Use parsed_sensor to compute bounds
        try:
            lb, ub = parsed_sensor_expr(func_source, input_bounds=bounds_combination)
            # Handle case where output is array
            # NOTE: is this necessary?
            if isinstance(lb, np.ndarray): 
                lb = float(lb[0])
            if isinstance(ub, np.ndarray):
                ub = float(ub[0])
            
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        except Exception as e:
            # HACK: this is a slightly jank way to handle this and is possibly unsafe given that I'm just eval-ing directly on the func_code
            # FIXME: make this less jank
            if isinstance(e, TypeError) and ("Output of the model is expected to be a single torch.Tensor. Actual type: <class 'int'>" in str(e) or "Output of the model is expected to be a single torch.Tensor. Actual type: <class 'float'>" in str(e)):
                try:
                    result = eval(func_code, {"np": np, "torch": torch}) # NOTE: may need to add more modules as necessary
                    return [(result, result)]
                except Exception as e:
                    raise Exception(f"Error computing CROWN bounds: {e}")
        
            raise Exception(f"Error computing CROWN bounds: {e}")
    
    # Return the minimum and maximum bounds
    return [(min(lower_bounds), max(upper_bounds))]

# Usage example:
# NOTE: consider making this it's own file, don't really want to be testing stuff in this file
# TODO: handle piecewise functions, may need to make some regularity assumptions (i.e., can't call pw function in a pw function) so that things make sense
if __name__ == "__main__":
    from verse.map import opendrive_map
    from enum import Enum, auto
    import os 

    class TrackMode(Enum):
        T0 = auto()
        T1 = auto()
        T2 = auto()
        M01 = auto()
        M12 = auto()
        M21 = auto()
        M10 = auto()
    
    def test_sensor(x, y):
        z = x**2 + y**2 # NOTE: CROWN just does this wrong apparently? min should be 0, not -0.5
        # z_safe = torch.sqrt(z)
        # xy_sum = torch.sum(torch.stack([x,y], dim=-1), dim=-1)
        w = arctan2(x,y)
        m = arctan2(y,x) # FIXME: currently these two return the same output, which is wrong
        # q = arctan2(x, w) # this works, but what this would represent in reality is unknown -- theoretically, w shouldn't be allowed as an input given this is from R^2 -> S1
        # l = w**2
        z = torch.sqrt(z) + torch.sum(torch.stack([x,y], dim=-1), dim=-1) # this is the correct way to do sums
        # # z = torch.sqrt(z) + x + y
        # w = arctan2(x, y)
        return z # currently this does nothing, a return function just needs to exist
    
    def map_sensor(x,y):
        # z = get_lane_heading(x,y)
        z = -2*np.pi
        return z

    def vis_sensor_piecewise(psi, phi):
        if psi >= 0 and psi <= phi:
            return -1
        elif psi < 0 and psi >= -phi:
            return 1
        elif psi > phi:
            # return -2 
            return -2-psi**2 # parser should be able to parse out arbitrary functions
        else:
            return 2
    
    def pw(psi, phi):
        y = vis_sensor_piecewise(psi, phi)
        return y
    # TODO: add a way to automatically create the input bounds given a list of bounds and the function header -- user shouldn't need to do this by hand
    # input_bounds = {'x': (0, 1), 'y': (1, 2)}
    # input_bounds = {'x': [(0, 1)], 'y': [(1, 2), (2,3)]}
    # input_bounds = {'x': [(-1, -0.5)], 'y': [(-1, 0.5)]} 
    
    # input_bounds = {'x': [(134, 134.01)], 'y': [(11.5, 11.51)]} # TODO: check to see if this is consistent with the stanley scenario
    # script_dir = os.path.realpath(os.path.dirname(__file__))
    # tmp_map = opendrive_map(os.path.join(script_dir, "t1_triple.xodr"))
    # final_bounds, history = parse_function(map_sensor, input_bounds, track_map=tmp_map, track_mode=TrackMode.T1)

    input_bounds = {'psi': [(-1.1, 2)], 'phi': [(1,1)]}
    final_bounds, history = parse_function(pw, input_bounds, [vis_sensor_piecewise]) # NOTE: is this the best way to handle piecewise functions? it's easy, but slightly unintuitive


    # final_bounds, history = parse_function(test_sensor, input_bounds)
    
    print("\nFinal bounds:")
    for var, bounds in final_bounds.items():
        for bound in bounds:
            print(f"  {var} ∈ [{bound[0]:.6f}, {bound[-1]:.6f}]")