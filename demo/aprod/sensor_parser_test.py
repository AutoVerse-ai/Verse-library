import ast
import inspect, textwrap
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from prox_error_all_bounds import box_extreme_error

FUNC_MAP = {
    "sin": "torch.sin",
    "cos": "torch.cos",
    # "atan2": "torch.atan2",
    "atan2": "atan2_crown",
    "norm": "norm",
}

# def norm(x, *args, **kwargs):
#     """
#     Wrapper around torch.norm that accepts either:
#     - a tensor, or
#     - a list/tuple of tensors (e.g. [x, y])
#     """
#     if isinstance(x, (list, tuple)):
#         # stack into a tensor along a new dimension
#         x = torch.stack(list(x))
#     return torch.norm(x, *args, **kwargs)

def norm(x, *args, **kwargs):
    """
    Replacement for torch.norm that avoids ReduceL2 (unsupported in CROWN).
    Works for:
      - a single tensor
      - a list/tuple of tensors (like [x, y])
    """
    if isinstance(x, (list, tuple)):
        x = torch.stack(list(x), dim=-1)
    # manually compute L2 norm: sqrt(sum(x^2))
    return torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=kwargs.get("keepdim", False)) + 1e-6)

def atan2_crown(y, x):
    eps = 1e-6 # some small number to keep things well-defined
    theta = torch.atan(y / (x + eps))

    # Approximate "x < 0" with relu
    x_neg = torch.relu(-x) / (torch.abs(x) + eps)  # ≈ 1 if x<0 else 0

    # Approximate "y < 0" with relu
    y_neg = torch.relu(-y) / (torch.abs(y) + eps)  # ≈ 1 if y<0 else 0

    # correction: +pi if x<0,y>=0 ; -pi if x<0,y<0
    correction = torch.pi * x_neg * (1 - 2*y_neg)

    return theta + correction


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
                         "norm": norm,
                         "atan2_crown": atan2_crown,
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

if __name__ == "__main__":

    def noisy_sensor(x, y, e_rho, e_theta):
        rho = norm([x, y])
        theta = atan2(y, x)
        rho_p = rho + e_rho
        theta_p = theta + e_theta
        return rho_p * cos(theta_p), rho_p * sin(theta_p)
        # return rho_p * cos(theta_p)

    model = TorchFuncModule(noisy_sensor)

    x = torch.tensor([1.0])
    y = torch.tensor([1.0])
    # e_rho = torch.tensor([0.05])
    # e_theta = torch.tensor([-0.02])
    e_rho = e_theta = torch.zeros(1)

    out = model(x, y, e_rho, e_theta)
    print(out)  # (tensor(...), tensor(...))

    dummy_x = torch.zeros(1, 1)       # batch size 1, dim 2
    dummy_y = torch.zeros(1, 1)
    dummy_e_rho = torch.zeros(1, 1)
    dummy_e_theta = torch.zeros(1, 1)

    lirpa_model = BoundedModule(model, (dummy_x, dummy_y, dummy_e_rho, dummy_e_theta), device="cpu")

    # Interval bounds for each input
    xl, xu = x-0.5, x+0.5
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

    