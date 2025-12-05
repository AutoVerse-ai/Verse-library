import ast
import inspect, textwrap
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from prox_error_all_bounds import box_extreme_error
import time
from multiprocessing import Pool
from itertools import product
import numpy as np
import pickle
import os
from pathlib import Path
import hashlib

FUNC_MAP = {
    "sin": "torch.sin",
    "cos": "torch.cos",
    # "atan2": "torch.atan2",
    # "atan2": "atan2_crown_safe",
    # "atan2": "atan2_crown",
    "norm": "norm",
}

def get_cache_key(input_bounds, function_name):
    """Generate a unique key for the input bounds and function name"""
    # Convert bounds to string and combine with function name
    bounds_str = str(input_bounds.tolist())
    combined_str = f"{bounds_str}_{function_name}"
    return hashlib.md5(combined_str.encode()).hexdigest()

def clear_sensor_cache():
    """Clear all cached sensor results"""
    cache_dir = Path(__file__).parent / "sensor_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("bounds_*.pkl"):
            f.unlink()

def norm(x, *args, **kwargs):
    """
    Replacement for torch.norm that avoids ReduceL2 (unsupported in CROWN).
    Works for:
      - a single tensor
      - a list/tuple of tensors (like [x, y])
    Ensures non-negative inputs for sqrt by using x*x instead of abs(x).
    """
    if isinstance(x, (list, tuple)):
        x = torch.stack(list(x), dim=-1)
    # Use x*x to ensure non-negativity - this keeps gradient information 
    # better than abs(x) or x**2
    squared = torch.sum(x * x, dim=-1, keepdim=kwargs.get("keepdim", False))
    # Add small epsilon to prevent divide by zero in backward pass
    return torch.sqrt(squared + 1e-8)

class TorchFuncModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._compiled = self._compile(fn)
    
    def _compile(self, fn):
        # get source code
        src = inspect.getsource(fn)
        src = textwrap.dedent(src) 
        tree = ast.parse(src)

        # function def is the first node
        func_def = tree.body[0]
        arg_names = [arg.arg for arg in func_def.args.args]

        # replace bare names with torch equivalents
        class TorchTransformer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in FUNC_MAP:
                    return ast.copy_location(
                        ast.parse(FUNC_MAP[node.id], mode="eval").body,
                        node
                    )
                return node

        new_tree = TorchTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)

        code = compile(new_tree, filename="<userfunc>", mode="exec")

        def wrapped(*args):
            local_env = {"torch": torch, 
                         "np": np,
                         "norm": norm,
                        #  "atan2_crown": atan2_crown,
                        #  "atan2_crown_safe": atan2_crown_safe,
                         }
            for name, val in zip(arg_names, args):
                local_env[name] = val
            exec(code, local_env)
            return local_env[func_def.name](*args)  # call transformed fn

        return wrapped

    def forward(self, *args):
        # return self._compiled(*args)
        out = self._compiled(*args)
        if isinstance(out, tuple) or isinstance(out, list):
            # return torch.stack(out, dim=-1).squeeze(1)
            return torch.concat(list(out), dim=-1) # shouldn't need list(out) but required due to torch quirks -- alternative above works without needing casting to list first 
        return out

# def compute_bounds_for_split(args):
#     """Helper function to compute bounds for a single split"""
#     split_bounds, sensor_function = args
    
#     # Create model inside worker process
#     model = TorchFuncModule(sensor_function)
#     dummy_inputs = tuple([torch.zeros(1, 1) for _ in split_bounds])
#     lirpa_model = BoundedModule(model, dummy_inputs, device="cpu")
    
#     bounded_inputs = []
#     for bounds in split_bounds:
#         lower, upper = bounds
#         lower = torch.tensor(lower, dtype=torch.float32)
#         upper = torch.tensor(upper, dtype=torch.float32)
#         center = ((lower + upper) / 2).unsqueeze(0)
#         perturb = PerturbationLpNorm(x_L=lower.unsqueeze(0), x_U=upper.unsqueeze(0))
#         bounded_inputs.append(BoundedTensor(center, perturb))
    
#     lb, ub = lirpa_model.compute_bounds(x=tuple(bounded_inputs), method="CROWN")
#     return lb.detach().numpy()[0], ub.detach().numpy()[0]
    
def parsed_sensor(sensor_function, inputs=None, input_bounds=None, device="cpu", sim: bool = False, num_splits=2):
    """
    Process inputs through a sensor function using TorchFuncModule with domain splitting
    
    Args:
        sensor_function: The sensor function to be processed
        inputs: For sim=True: numpy array of inputs. Ignored if sim=False.
        input_bounds: List of (lower, upper) bound tuples for each input. Required if sim=False
        device: Device to run computations on
        sim: If True, run in simulation mode. If False, compute bounds
        num_splits: Number of splits per dimension (default=2)
    """

    if sim:
        # ... existing simulation code ...
        if inputs is None:
            raise ValueError("inputs required for simulation mode")
        torch_inputs = [torch.tensor([x], dtype=torch.float32) for x in inputs]
        return model(*torch_inputs).numpy()
    
    if input_bounds is None:
        raise ValueError("input_bounds required for bound computation mode")
    
    # Create cache directory if it doesn't exist
    cache_dir = Path(__file__).parent / "sensor_cache" # this creates the folder at aprod, can change later to just Path("--") if I'd rather create the folder in the working directory instead
    cache_dir.mkdir(exist_ok=True)
    
    # Generate cache key from input bounds and function name
    cache_key = get_cache_key(np.array(input_bounds), sensor_function.__name__)
    cache_file = cache_dir / f"bounds_{cache_key}_splits_{num_splits}.pkl"
    
    # Check if cached result exists
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
        
    model = TorchFuncModule(sensor_function)
    dummy_inputs = tuple([torch.zeros(1, 1) for _ in input_bounds])
    lirpa_model = BoundedModule(model, dummy_inputs, device=device)
    
    # Initialize bounds as None
    global_lb = None
    global_ub = None
    
    # Generate split points for each dimension
    splits = []
    for lower, upper in input_bounds:
        if lower == upper:
            # For unsplit dimensions, create a list with a single tuple
            splits.append([(lower, lower)])
        else:
            split_points = np.linspace(lower, upper, num_splits+1)
            # Create list of tuples for split points
            splits.append(list(zip(split_points[:-1], split_points[1:])))
    
    # Compute cartesian product of splits
    for split_bounds in product(*splits):
        # Create bounded tensors for this sub-domain
        bounded_inputs = []
        for bounds in split_bounds:  # bounds is already a (lower, upper) tuple
            lower, upper = bounds  # Unpack the tuple
            lower = torch.tensor(lower, dtype=torch.float32)
            upper = torch.tensor(upper, dtype=torch.float32)
            center = ((lower + upper) / 2).unsqueeze(0)
            lower = lower.unsqueeze(0)
            upper = upper.unsqueeze(0)
            perturb = PerturbationLpNorm(x_L=lower, x_U=upper)
            bounded_inputs.append(BoundedTensor(center, perturb))
        
        # Compute bounds for this sub-domain
        lb, ub = lirpa_model.compute_bounds(
            x=tuple(bounded_inputs),
            method="CROWN"
        )
        
        # Update global bounds
        lb, ub = lb.detach().numpy()[0], ub.detach().numpy()[0]
        if global_lb is None:
            global_lb = lb
            global_ub = ub
        else:
            global_lb = np.minimum(global_lb, lb)
            global_ub = np.maximum(global_ub, ub)
    
    with open(cache_file, 'wb') as f:
        pickle.dump((global_lb, global_ub), f)

    return global_lb, global_ub
    # all_splits = list(product(*splits))
    
    # # Create pool of workers
    # with Pool() as pool:
    #     results = pool.map(compute_bounds_for_split,
    #                      [(split, sensor_function) for split in all_splits])
    
    # # Combine results
    # global_lb = np.minimum.reduce([lb for lb, _ in results])
    # global_ub = np.maximum.reduce([ub for _, ub in results])
    
    # return global_lb, global_ub

if __name__ == "__main__":
    import numpy as np
    
    start = time.perf_counter()
    clear_sensor_cache()
    
    # input_bounds = np.array([[0, 0], [0,0], [-5, 5], [-2, 2], [0, 0], [1, 1]])
    input_bounds = np.array([[-1,-1], [-1, 1]])
    # input_bounds = np.array([[0,0], [0,0,], [0.001, 0.001], [1, 1]])
    # lb, ub = parsed_sensor(atan2_test, input_bounds=input_bounds, num_splits=2)
    lb, ub = parsed_sensor_with_atan2_partitioning(atan2_test_only, input_bounds=input_bounds, num_splits=2)
    # lb, ub = parsed_sensor_with_atan2_partitioning(cur_sense_relu, input_bounds=input_bounds, num_splits=2)

    print(lb, ub, f'Runtime: {time.perf_counter()-start:.2f} s')