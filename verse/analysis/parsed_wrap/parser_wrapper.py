import ast
import inspect
import textwrap
from typing import Callable, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from .sensor_parser import parsed_sensor_expr
from .wrapper_consts import ALIASES, ANGULAR_FUNCTIONS, MAP_FUNCTIONS
from .bounded_angular import angular_span_rect_parser, angular_bounds_diff_correct, combine_angular_bounds
from .bounded_map import get_heading_bounds_optimized, get_lateral_distance_bounds_optimized
import itertools
from .bounded_piecewise import bounded_piecewise_z3, parse_piecewise_function
import pickle
import os
from pathlib import Path
import hashlib

def get_cache_key(input_bounds, function_name):
    """Generate a unique key for the input bounds and function name using MD5 hash."""
    bounds_str = str(input_bounds)
    combined_str = f"{bounds_str}_{function_name}"
    return hashlib.md5(combined_str.encode()).hexdigest()

def clear_parse_cache():
    """Clear all cached parse_function results."""
    cache_dir = Path(__file__).parent / "parse_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("parse_*.pkl"):
            f.unlink()

def parse_function_array(func: Callable, input_bounds: List[List[float]], piecewise_functions: List[Callable] = None, track_mode=None, track_map=None, cache: bool = True, logging: bool = False, num_splits: int = 1):
    """
    Wrapper for parse_function that accepts array-like bounds and converts them to the required dictionary format before processing using `parse_function`.
    Supports domain partitioning for tighter bounds.
    
    Args:
        func: Function to parse (must contain only assignments and return)
        input_bounds: List of [min, max] pairs for each argument, e.g., [[min_x, max_x], [min_y, max_y], ...]
        piecewise_functions: List of piecewise functions (optional)
        track_mode: Track mode (optional)
        track_map: Track map (optional)
        cache: Whether to cache results (default True)
        num_splits: Number of splits per dimension (default 1, no partitioning)
    
    Returns:
        unioned_bounds: dict of arg_name-bound pairs
    
    Notes
    -----
        Caching is enabled by default because `verify` typically calls sense many times with the same arguments. 
        
    """

    # NOTE: Create cache directory if it doesn't exist in the verse/analysis/parsed_wrap directory
    # NOTE: Consider allow users to specify path as well
    cache_dir = Path(__file__).parent / "parse_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # NOTE: Extracting arg names for later
    sig = inspect.signature(func)
    arg_names = list(sig.parameters.keys())
    
    # NOTE: bound-argument size validation
    if len(input_bounds) != len(arg_names):
        raise ValueError(f"Number of bounds ({len(input_bounds)}) must match number of function arguments ({len(arg_names)})")
    
    # NOTE: bound validation
    for i, bounds in enumerate(input_bounds):
        if len(bounds) != 2:
            raise ValueError(f"Each bound must be a [min, max] pair, got {bounds} for argument {arg_names[i]}")
        min_val, max_val = bounds
        if min_val > max_val:
            raise ValueError(f"For argument {arg_names[i]}, expected max_val to be greater than min_val, got {min_val} (min_val) > {max_val} (max_val)")
    
    # NOTE: Create dict format for cache key
    parsed_input_bounds_for_key = {}
    for name, bounds in zip(arg_names, input_bounds):
        parsed_input_bounds_for_key[name] = [(bounds[0], bounds[1])]
    
    # NOTE: Generate cache key from input bounds and function name
    cache_key = get_cache_key(parsed_input_bounds_for_key, func.__name__)
    cache_file = cache_dir / f"parse_{cache_key}.pkl"
    
    # NOTE: Check if cached result exists
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            unioned_bounds, _ = pickle.load(f)
            return unioned_bounds
    
    if num_splits == 1:
        parsed_input_bounds = {}
        for name, bounds in zip(arg_names, input_bounds):
            min_val, max_val = bounds
            parsed_input_bounds[name] = [(min_val, max_val)]
        return parse_function(func, parsed_input_bounds, piecewise_functions, track_mode, track_map, cache, logging)
    
    # NOTE: Domain partitioning: generate splits for each dimension if bounds are non-trivial
    splits = []
    for bounds in input_bounds:
        min_val, max_val = bounds
        if min_val == max_val:
            splits.append([(min_val, max_val)])
        else:
            split_points = np.linspace(min_val, max_val, num_splits + 1)
            splits.append(list(zip(split_points[:-1], split_points[1:]))) # NOTE: Cleverly creating split bounds
    
    all_current_bounds = []
    # FIXME: Consider getting rid of all_bounds_history in this case or handle it better -- currently is worthless
    all_bounds_histories = []
    
    for split_bounds in itertools.product(*splits):
        parsed_input_bounds = {}
        for name, bounds in zip(arg_names, split_bounds):
            parsed_input_bounds[name] = [(bounds[0], bounds[1])]
        current_bounds, bounds_history = parse_function(func, parsed_input_bounds, piecewise_functions, track_mode, track_map, cache, parent_cache=True)
        all_current_bounds.append(current_bounds)
        all_bounds_histories.append(bounds_history)
    
    unioned_bounds = {}
    for var in all_current_bounds[0].keys():
        all_var_bounds = [cb[var] for cb in all_current_bounds]
        has_multiple = any(len(bounds) > 1 for bounds in all_var_bounds)
        # NOTE: Currently considering any variable with multiple bounds necessarily an angular bound
        if has_multiple:
            unioned_bounds[var] = combine_angular_bounds(all_var_bounds)
        else:
            all_mins = [b[0][0] for b in all_var_bounds]
            all_maxs = [b[0][1] for b in all_var_bounds]
            unioned_bounds[var] = [(min(all_mins), max(all_maxs))]
    
    if cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((unioned_bounds, None), f)

    return unioned_bounds

def parse_function(func: Callable, input_bounds: Dict[str, List[Tuple[float, float]]], piecewise_functions: List[Callable] = None, track_mode = None, track_map = None, cache: bool = True, logging: bool = False, parent_cache = False):
    """
    Process a function line-by-line, computing bounds at each step using parsed_sensor. Typically called as a helper function by `parsed_function_array`.
    
    Args:
        func: Function to analyze (must contain only assignments and return)
        input_bounds: Dict mapping variable names to (min, max) tuples
    
    Returns:
        (current_bounds, bounds_history): Tuple where current_bounds is a dict mapping arg_names to bounds, which is a list of bounds, and bounds_history, which is a list that contains the entire revision history of current_bounds.
    """
    cache_dir = Path(__file__).parent / "parse_cache"
    cache_dir.mkdir(exist_ok=True)
    
    cache_key = get_cache_key(input_bounds, func.__name__)
    cache_file = cache_dir / f"parse_{cache_key}.pkl"
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
        
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    func_def = tree.body[0]
    
    current_bounds = dict(input_bounds)
    bounds_history = [dict(input_bounds)]
    
    for stmt in func_def.body:
        # NOTE: Currently returns don't do anything aside from just ending the parse loop
        if isinstance(stmt, ast.Return):
            break
        
        elif isinstance(stmt, ast.Assign):
            # NOTE: Currently only handling single variable assignments in functions
            # TODO: Consider adding support for lists and list decomposition
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only simple assignments supported")
            
            var_name = stmt.targets[0].id
            rhs_expr = stmt.value
            rhs_code = ast.unparse(rhs_expr)
            
            rhs_bounds = compute_expression_bounds_with_parsed_sensor(
                rhs_code,
                list(current_bounds.keys()),
                current_bounds,
                piecewise_functions,
                track_mode,
                track_map
            )
            
            current_bounds[var_name] = rhs_bounds
            bounds_history.append(dict(current_bounds))
            
            if logging:
                print(f"After {var_name} = {rhs_code}")
                for rhs_bound in rhs_bounds:
                    print(f"  {var_name} ∈ [{rhs_bound[0]:.6f}, {rhs_bound[-1]:.6f}]")

        else:
            raise Exception(f'Line is of type {type(stmt)} instead of assign or return.')
    
    if cache and not parent_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump((current_bounds, bounds_history), f)

    return current_bounds, bounds_history


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
        List of (lower_bound, upper_bound) tuples
    
    Notes
    -----

    """
    for alias, func_name in ALIASES.items():
        if alias in expr_code:
            expr_code = expr_code.replace(alias, func_name)

    tree = ast.parse(expr_code)
    func_def = tree.body[0].value

    args = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call):
            args = [arg.id for arg in node.args if isinstance(arg, ast.Name) and arg.id in arg_names]
            break
    
    # NOTE: Currently checking type of function using function name exclusively
    if any(func in expr_code for func in ANGULAR_FUNCTIONS):
        func_name = [func for func in ANGULAR_FUNCTIONS if func in expr_code][0]
        bounds = handle_angular_function(func_name, args, current_bounds)
        return bounds
    
    elif piecewise_functions is not None and any(func.__name__ in expr_code for func in piecewise_functions):
        func = next(func for func in piecewise_functions if func.__name__ in expr_code)
        bounds = handle_piecewise_function(func, args, current_bounds)
        return bounds
    
    elif any(func in expr_code for func in MAP_FUNCTIONS):
        func_name = [func for func in MAP_FUNCTIONS if func in expr_code][0]
        bounds = handle_map_function(func_name, args, current_bounds, track_map, track_mode)
        return bounds
    else:
        # NOTE: If no match to handle function in a special way, default to handling function as a CROWN function.
        bounds = handle_crown_function(expr_code, arg_names, current_bounds)
        return bounds


def handle_angular_function(func_name: str, args: List[str], current_bounds: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
    """
    Handle angular functions (e.g., atan2, arctan2).
    
    Args:
        func_name: Name of the angular function
        args: List of argument names
        current_bounds: Current variable bounds
    
    Returns:
        List of (lower_bound, upper_bound) tuples
    """
    func = globals()[func_name]
    
    bounds_list = [current_bounds[name] for name in args]
    output_bounds = []
    
    for bounds_combination in itertools.product(*bounds_list):
        try:
            bounds = func(*bounds_combination)
            output_bounds.append(bounds)
        except Exception as e:
            raise Exception(f"Error computing angular bounds: {e}")
    
    combined_bounds = combine_angular_bounds(output_bounds)
    
    return combined_bounds


def handle_map_function(func_name: str, args: List[str], current_bounds: Dict[str, List[Tuple[float, float]]], track_mode, track_map) -> Tuple[float, float]:
    """
    Handle map functions that require track_mode and track_map.
    
    Args:
        func_name: Name of the map function
        args: List of argument names
        current_bounds: Current variable bounds
    
    Returns:
        List of (lower_bound, upper_bound) tuples
    """
    func = globals()[func_name]
    
    bounds_list = [current_bounds[name] for name in args]
    output_bounds = []

    for bounds_combination in itertools.product(*bounds_list):
        try:
            bounds = func(track_mode, track_map, *bounds_combination)
            output_bounds.append(bounds)
        except Exception as e:
            raise Exception(f"Error computing map function bounds: {e}")
    
    return [(min(np.array(output_bounds)[:,0]), max(np.array(output_bounds)[:,1]))]

def handle_piecewise_function(func: Callable, args: List[str], current_bounds: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
    """
    Handle piecewise functions using Z3 analysis.
    
    Args:
        func: Piecewise function to analyze
        args: List of argument names
        current_bounds: Current variable bounds
    
    Returns:
        List of (lower_bound, upper_bound) tuples
    """
    bounds_list = [current_bounds[name] for name in args] 
    all_output_bounds = []
    # NOTE: Use helper to first convert piecewise function to (z3 expression, output expression) pairs 
    parsed_func = parse_piecewise_function(func)

    for bounds_combination in itertools.product(*bounds_list):
        try:
            bounds_dict = dict(zip(args, bounds_combination))
            res = bounded_piecewise_z3(parsed_func, bounds_dict)
            for i in range(len(res)):
                output_function = str(res[i][1])
                # NOTE: Using constrained bounds so that if only part of the argument's bounds satisfy an expression, only that part will be used
                constrained_bounds = {arg: [res[i][0][arg]] for arg in res[i][0]}
                # NOTE: Assuming piecewise functions don't recursively call piecewise functions.
                output_bounds = compute_expression_bounds_with_parsed_sensor(
                    output_function,
                    args,
                    constrained_bounds,
                )
                all_output_bounds.extend(output_bounds) 

        except Exception as e:
            raise Exception(f"Error computing piecewise function bounds: {e}")

    min_bound = min(min(output_bound) for output_bound in all_output_bounds)
    max_bound = max(max(output_bound) for output_bound in all_output_bounds)

    return [(min_bound, max_bound)]

def handle_crown_function(func_code: str, arg_names: List[str], current_bounds: Dict[str, List[Tuple[float, float]]]) -> Tuple[float, float]:
    """
    Handle general expressions using CROWN (auto_LiRPA) for bound computation.
    
    Args:
        func_code: Code of the RHS expression
        arg_names: Complete list of argument names
        current_bounds: Current variable bounds
    
    Returns:
        List of (lower_bound, upper_bound) tuples
    """
    tree = ast.parse(func_code)
    func_def = tree.body[0].value

    args = []
    for node in ast.walk(func_def):
        if isinstance(node, ast.Name) and node.id in arg_names:
            args.append(node.id)

    args = sorted(list(set(args)))

    func_source = f"""
def temp_func({', '.join(args)}):
    return {func_code}
"""
    
    bounds_list = [current_bounds[name] for name in args]
    
    lower_bounds = []
    upper_bounds = []
    
    for bounds_combination in itertools.product(*bounds_list):
        try:
            lb, ub = parsed_sensor_expr(func_source, input_bounds=bounds_combination)
            if isinstance(lb, np.ndarray): 
                lb = float(lb[0])
            if isinstance(ub, np.ndarray):
                ub = float(ub[0])
            
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        except Exception as e:
            if isinstance(e, TypeError) and ("Output of the model is expected to be a single torch.Tensor. Actual type: <class 'int'>" in str(e) or "Output of the model is expected to be a single torch.Tensor. Actual type: <class 'float'>" in str(e)):
                try:
                    result = eval(func_code, {"np": np, "torch": torch})
                    return [(result, result)]
                except Exception as e:
                    raise Exception(f"Error computing CROWN bounds: {e}")
        
            raise Exception(f"Error computing CROWN bounds: {e}")
    
    return [(min(lower_bounds), max(upper_bounds))]

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
        z = x**2 + y**2
        w = arctan2(x,y)
        m = arctan2(y,x)
        z = torch.sqrt(z) + torch.sum(torch.stack([x,y], dim=-1), dim=-1)
        return z
    
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
            return -2-psi**2
        else:
            return 2
    
    def pw(psi, phi):
        y = vis_sensor_piecewise(psi, phi)
        return y

    input_bounds = {'psi': [(-1.1, 2)], 'phi': [(1,1)]}
    final_bounds, history = parse_function(pw, input_bounds, [vis_sensor_piecewise])
    
    print("\nFinal bounds:")
    for var, bounds in final_bounds.items():
        for bound in bounds:
            print(f"  {var} ∈ [{bound[0]:.6f}, {bound[-1]:.6f}]")

""" 
NOTE: The most complex part requiring some attention is the handling of piecewise functions
and the interaction between Z3 optimization (bounded_piecewise_z3) and CROWN-based bound
propagation. The compute_expression_bounds_with_parsed_sensor function dispatches to different
handlers based on function type, which can create subtle edge cases when functions are nested.
"""