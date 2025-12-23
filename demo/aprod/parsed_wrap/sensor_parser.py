import ast
import inspect, textwrap
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from .bounded_angular import box_extreme_error
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
    "atan2": "atan2_crown_safe",
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

def atan2_crown(y, x):
    """
    Compute atan2 in a way compatible with CROWN bound computation.
    Avoids negative denominators by using alternative formulation.
    """
    eps = 1e-6  # small number for numerical stability
    
    # We can rearrange y/x into sign(x)*y / abs(x) to ensure denominator is positive
    abs_x = torch.abs(x)  #
    y_adj = y * torch.sign(x + eps)
    
    # Now compute arctan with guaranteed positive denominator
    theta = torch.atan(y_adj / abs_x)

    # Handle quadrants correctly
    x_neg = torch.relu(-x) / (abs_x + eps)  # ≈ 1 if x<0 else 0
    y_neg = torch.relu(-y) / (torch.abs(y)+eps)  # ≈ 1 if y<0 else 0

    # Quadrant corrections
    correction = torch.pi * x_neg * (1 - 2*y_neg)
    
    return theta + correction

def atan2_crown_safe(y, x):
    """
    Compute atan2 assuming x and y bounds don't cross zero.
    """
    eps = 1e-8
    
    abs_x = torch.abs(x)  #
    y_adj = y * torch.sign(x + eps)
    
    # Now compute arctan with guaranteed positive denominator
    theta = torch.atan(y_adj / abs_x)
    
    # Correction using only sign(), constants, and arithmetic
    # When x < 0: correction = π * sign(y)
    # When x > 0: correction = 0
    # 
    # This can be written as:
    # correction = π * (1 - sign(x)) / 2 * sign(y)
    #
    # Derivation:
    # sign(x) = +1 if x > 0, -1 if x < 0
    # (1 - sign(x))/2 = (1 - 1)/2 = 0 if x > 0
    #                 = (1 - (-1))/2 = 1 if x < 0
    correction = torch.pi * (1 - torch.sign(x)) / 2 * torch.sign(y)
    
    return theta + correction

# def atan2_crown(y, x): # less accurate but more operations -> worse performance
#     """
#     Ultra-simplified linear approximation of atan2.
#     Avoids division by sqrt to prevent Auto-LiRPA's special case handling.
#     """
#     eps = 1e-6
    
#     # Instead of y/sqrt(x^2), compute y*x/sqrt(x^2 * x^2)
#     x_sq = x * x
#     numerator = y * x
#     denom = torch.sqrt(x_sq * x_sq + eps)
    
#     # Basic ratio computation
#     theta = numerator / denom
    
#     # Quadrant correction using relu
#     x_is_neg = torch.relu(-x)
    
#     return theta + (x_is_neg * torch.pi)

class TorchFuncModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self._compiled = self._compile(fn)
    
    def _compile(self, fn):
        # get source code
        # src = inspect.getsource(fn)
        # src = textwrap.dedent(src) 

        src: str = None
        if type(fn) is str:
            src = fn # assume that we are grabbing directly from code
        else: # FIXME: this should be an elif
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
                         "atan2_crown": atan2_crown,
                         "atan2_crown_safe": atan2_crown_safe,
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

# def parsed_sensor(sensor_function, inputs=None, input_bounds=None, device="cpu", sim: bool = False):
#     """
#     Process inputs through a sensor function using TorchFuncModule
    
#     Args:
#         sensor_function: The sensor function to be processed
#         inputs: For sim=True: numpy array of inputs. Ignored if sim=False.
#         input_bounds: List of (lower, upper) bound tuples for each input. Required if sim=False
#         device: Device to run computations on
#         sim: If True, run in simulation mode. If False, compute bounds
    
#     Returns:
#         If sim=True: Direct sensor output
#         If sim=False: (lower bounds, upper bounds) of the sensor output
#     """
#     model = TorchFuncModule(sensor_function)
    
#     if sim:
#         if inputs is None:
#             raise ValueError("inputs required for simulation mode")
#         # Convert numpy inputs to torch tensors
#         torch_inputs = [torch.tensor([x], dtype=torch.float32) for x in inputs]
#         return model(*torch_inputs).numpy()
#     else:
#         if input_bounds is None:
#             raise ValueError("input_bounds required for bound computation mode")
            
#         # Create dummy inputs for CROWN model initialization
#         dummy_inputs = tuple([torch.zeros(1, 1) for _ in input_bounds])
        
#         # Initialize CROWN model
#         lirpa_model = BoundedModule(model, dummy_inputs, device=device)
        
#         # Create bounded tensors for each input
#         bounded_inputs = []
#         for lower, upper in input_bounds:
#             # Convert to torch tensors if they're numpy arrays
#             lower = torch.tensor(lower, dtype=torch.float32)
#             upper = torch.tensor(upper, dtype=torch.float32)
            
#             # Calculate center as average of bounds
#             center = ((lower + upper) / 2).unsqueeze(0)
#             lower = lower.unsqueeze(0)
#             upper = upper.unsqueeze(0)
            
#             # Create perturbation and bounded tensor
#             perturb = PerturbationLpNorm(x_L=lower, x_U=upper)
#             bounded_inputs.append(BoundedTensor(center, perturb))
        
#         # Compute bounds using CROWN
#         lb, ub = lirpa_model.compute_bounds(
#             x=tuple(bounded_inputs),
#             method="CROWN"
#         )
        
#         return lb.detach().numpy()[0], ub.detach().numpy()[0]

def compute_bounds_for_split(args):
    """Helper function to compute bounds for a single split"""
    split_bounds, sensor_function = args
    
    # Create model inside worker process
    model = TorchFuncModule(sensor_function)
    dummy_inputs = tuple([torch.zeros(1, 1) for _ in split_bounds])
    lirpa_model = BoundedModule(model, dummy_inputs, device="cpu")
    
    bounded_inputs = []
    for bounds in split_bounds:
        lower, upper = bounds
        lower = torch.tensor(lower, dtype=torch.float32)
        upper = torch.tensor(upper, dtype=torch.float32)
        center = ((lower + upper) / 2).unsqueeze(0)
        perturb = PerturbationLpNorm(x_L=lower.unsqueeze(0), x_U=upper.unsqueeze(0))
        bounded_inputs.append(BoundedTensor(center, perturb))
    
    lb, ub = lirpa_model.compute_bounds(x=tuple(bounded_inputs), method="CROWN")
    return lb.detach().numpy()[0], ub.detach().numpy()[0]
    
def parsed_sensor(sensor_function, inputs=None, input_bounds=None, device="cpu", sim: bool = False, num_splits=1):
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

def parsed_sensor_expr(sensor_src, input_bounds, device = 'cpu', nonce='', num_splits=1):
    if input_bounds is None:
        raise ValueError("input_bounds required for bound computation mode")
        
    # TODO: start caching only when nonce exists for each expression     
    # # Create cache directory if it doesn't exist
    # cache_dir = Path(__file__).parent / "sensor_cache" # this creates the folder at aprod, can change later to just Path("--") if I'd rather create the folder in the working directory instead
    # cache_dir.mkdir(exist_ok=True)
    
    # # Generate cache key from input bounds and function name
    # cache_key = get_cache_key(np.array(input_bounds), nonce)
    # cache_file = cache_dir / f"bounds_{cache_key}_splits_{num_splits}.pkl"
    
    # # Check if cached result exists
    # if cache_file.exists():
    #     with open(cache_file, 'rb') as f:
    #         return pickle.load(f)
        
    model = TorchFuncModule(sensor_src)
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
    
    # with open(cache_file, 'wb') as f:
    #     pickle.dump((global_lb, global_ub), f)

    return global_lb, global_ub


if __name__ == "__main__":
    import numpy as np

    def noisy_sensor(x, y, e_rho, e_theta):
        rho = norm([x, y])
        theta = atan2(y, x)
        rho_p = rho + e_rho
        theta_p = theta + e_theta
        return rho_p * cos(theta_p), rho_p * sin(theta_p)

    def prox_error_ver(x, y, z, ep_r, ep_ang):
        rho = norm([x,y,z])
        rho_plane = norm([x,y])
        theta = atan2(y,x)
        psi = atan2(z, rho_plane)
        rho_p, theta_p, psi_p = rho+ep_r, theta+ep_ang, psi+ep_ang
        nx, ny, nz = rho_p*cos(theta_p)*cos(psi_p), rho_p*sin(theta_p)*cos(psi_p), rho_p*sin(psi_p)
        return x-nx, y-ny, z-nz, cos(x)**1.33
    
    def cur_sense_relu(x, y, other_x, other_y, theta, phi):
        """
        Convert conditional cur_sense logic to ReLU-based function
        
        Args:
            rel_x, rel_y: relative position to assigned agent
            theta: ego agent heading
            phi: sensor field of view half-angle
        
        Returns:
            cur_sense value: -2, -1, 1, or 2
        """
        # Step 1: Compute psi = normalized relative angle
        rel_x, rel_y = other_x - x, other_y-y
        psi = ((atan2(rel_y, rel_x) - theta + np.pi) % (2*np.pi)) - np.pi

        eps = 1e-6

        out_left = torch.relu(psi-phi+eps) # > 0 for any region implies that the sensor will output that value
        right = (torch.relu(psi+phi)*torch.relu(-psi+eps))
        left = (torch.relu(psi)*torch.relu(phi-psi))
        # region3 = (torch.relu(psi)*torch.relu(phi-psi))
        out_right = torch.relu(-phi-psi+eps)
        # Combine regions with their output values
        # cur_sense = -region3
        # cur_sense = rel_x*0

        return out_left, left, right, out_right, psi # Return both for debugging/validation
        
    def vis_sense_cont(x,y,other_x, other_y, theta, phi):
        """
        Outputs continuous y, if 0<y<1, then right, y>1 out right, -1<y<0 left, y<-1 out left -- can do this DL
        Issues with discontinuity around pi/-pi -- but note that y>>0 in this case, can try to handle separately
        """
        rel_x, rel_y = other_x - x, other_y-y
        # psi = ((atan2(rel_y, rel_x) - theta + np.pi) % (2*np.pi)) - np.pi
        psi = atan2(rel_y, rel_x) - theta
        return psi/phi

    def atan2_test(x, y, other_x, other_y):
        eps = 1e-6  # small number for numerical stability
    
        rel_x, rel_y = other_x-x, other_y-y
        abs_x = torch.abs(rel_x)  #
        y_adj = rel_y * torch.sign(rel_x + eps)
        
        # Now compute arctan with guaranteed positive denominator
        theta = torch.atan(y_adj / (abs_x+eps))

        # Handle quadrants correctly
        # x_neg = torch.relu(-rel_x) / (abs_x + eps)  # ≈ 1 if x<0 else 0
        x_neg = torch.sigmoid(-rel_x*100) 
        # y_neg = torch.relu(-rel_y) / (torch.abs(rel_y)+eps)  # ≈ 1 if y<0 else 0
        y_neg = torch.sigmoid(-rel_y*100)

        # Quadrant corrections
        correction = torch.pi * x_neg * (1 - 2*y_neg)
        
        return theta+correction
    
    def atan2_test_only(x,y):
        return atan2(y, x)
    # model = TorchFuncModule(noisy_sensor)

    # inputs = np.array([1,1,0,0])
    # out = parsed_sensor(noisy_sensor, inputs, sim=True)
    # print(out)
    
    start = time.perf_counter()
    clear_sensor_cache()

    # input_bounds = [(.95,1.05), (1,1), (0,0), (0,0)]
    # input_bounds = np.array([[.95,1.05], [1,1], [0,0], [0,0], [0,0]])
    # input_bounds = np.array([[-5,-3], [-2,-1], [0,0], [-0.01,0.01], [-1e-6,1e-6]])
    # input_bounds = np.array([[-5,-3], [-2,-1], [0,0], [0,0], [0,0]])
    # lb, ub = parsed_sensor(prox_error_ver, input_bounds=input_bounds, num_splits=2)
    
    # input_bounds = np.array([[0, 0], [0,0], [-1,1], [1,1], [0,0], [1,1]]) # test out more combinations
    # input_bounds = np.array([[0, 0], [0,0], [-10,-0.05], [1,1]]) # 
    # lb, ub = parsed_sensor(atan2_test, input_bounds=input_bounds, num_splits=4)
    # lb, ub = parsed_sensor(cur_sense_relu, input_bounds=input_bounds, num_splits=3)
    # lb, ub = parsed_sensor(vis_sense_cont, input_bounds=input_bounds, num_splits=3)
    
    # input_bounds = np.array([[0, 0], [0,0], [-5, 5], [-2, 2], [0, 0], [1, 1]])
    input_bounds = np.array([[-1,-1], [-1, 1]])
    # input_bounds = np.array([[0,0], [0,0,], [0.001, 0.001], [1, 1]])
    # lb, ub = parsed_sensor(atan2_test, input_bounds=input_bounds, num_splits=2)
    lb, ub = parsed_sensor_with_atan2_partitioning(atan2_test_only, input_bounds=input_bounds, num_splits=2)
    # lb, ub = parsed_sensor_with_atan2_partitioning(cur_sense_relu, input_bounds=input_bounds, num_splits=2)

    print(lb, ub, f'Runtime: {time.perf_counter()-start:.2f} s')
    exit()

    x = torch.tensor([1.0])
    y = torch.tensor([1.0])
    # e_rho = torch.tensor([0.05])
    # e_theta = torch.tensor([-0.02])
    e_rho = e_theta = torch.zeros(1)

    out = model(x, y, e_rho, e_theta)
    # print(out)  # (tensor(...), tensor(...))

    dummy_x = torch.zeros(1, 1)       # batch size 1, dim 2
    dummy_y = torch.zeros(1, 1)
    dummy_e_rho = torch.zeros(1, 1)
    dummy_e_theta = torch.zeros(1, 1)

    lirpa_model = BoundedModule(model, (dummy_x, dummy_y, dummy_e_rho, dummy_e_theta), device="cpu")

    # Interval bounds for each input
    xl, xu = x-0.05, x+0.05
    yl, yu = y, y
    e_rho_l, e_rho_u = e_rho, e_rho    # example
    e_theta_l, e_theta_u = e_theta, e_theta

    # Compute centers with batch dim
    x_center = ((xl + xu) / 2).unsqueeze(0)          # shape (1,2)
    y_center = ((yl + yu) / 2).unsqueeze(0)
    e_rho_center = ((e_rho_l + e_rho_u) / 2).unsqueeze(0)
    e_theta_center = ((e_theta_l + e_theta_u) / 2).unsqueeze(0)

    # Perturbations
    x_perturb = PerturbationLpNorm(x_L=xl.unsqueeze(0), x_U=xu.unsqueeze(0))
    y_perturb = PerturbationLpNorm(x_L=yl.unsqueeze(0), x_U=yu.unsqueeze(0))
    e_rho_perturb = PerturbationLpNorm(x_L=e_rho_l.unsqueeze(0), x_U=e_rho_u.unsqueeze(0))
    e_theta_perturb = PerturbationLpNorm(x_L=e_theta_l.unsqueeze(0), x_U=e_theta_u.unsqueeze(0))

    # Bounded tensors
    x_bounded = BoundedTensor(x_center, x_perturb)
    y_bounded = BoundedTensor(y_center, y_perturb)
    e_rho_bounded = BoundedTensor(e_rho_center, e_rho_perturb)
    e_theta_bounded = BoundedTensor(e_theta_center, e_theta_perturb)

    # Compute bounds
    lb, ub = lirpa_model.compute_bounds(
        x=(x_bounded, y_bounded, e_rho_bounded, e_theta_bounded),
        method="CROWN"  # or "CROWN"
    ) 
    print(lb, ub) # note that after some course testing, this seems to be a valid overapproximation -- though do note it's pretty coarse

    # note that what box_extreme_error is trying to do is find bounds on the error, not the estimate itself, so need to change function before doing this again
    # x_bounds_opt, y_bounds_opt = box_extreme_error([(x.item()-0.1, x.item()+0.1), (y.item(), y.item()), (0,0)], 0, 0, 'x'), box_extreme_error([(x.item()-0.1, x.item()+0.1), (y.item(), y.item()), (0,0)], 0, 0, 'y')

