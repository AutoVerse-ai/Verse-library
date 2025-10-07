import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Model: squared Euclidean norm
class SquaredNormDiff(torch.nn.Module):
    def forward(self, x, y):
        diff = x - y               # shape (batch, dim)
        return torch.sum(diff * diff, dim=1, keepdim=True)  # shape (batch, 1)

# Dummy inputs: batch=1, dim=2
dummy_x = torch.zeros(1, 2)
dummy_y = torch.zeros(1, 2)

lirpa_model = BoundedModule(SquaredNormDiff(), (dummy_x, dummy_y), device="cpu")

# Interval bounds
xl, xu = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])   # x ∈ [xl, xu]
yl, yu = torch.tensor([2.0, 0.0]), torch.tensor([3.0, 1.0])   # y ∈ [yl, yu]

# Centers with batch dimension
x_center = ((xl + xu) / 2).unsqueeze(0)  # shape (1,2)
y_center = ((yl + yu) / 2).unsqueeze(0)

# Interval perturbations (also need batch dim!)
x_perturb = PerturbationLpNorm(x_L=xl.unsqueeze(0), x_U=xu.unsqueeze(0))
y_perturb = PerturbationLpNorm(x_L=yl.unsqueeze(0), x_U=yu.unsqueeze(0))

x_bounded = BoundedTensor(x_center, x_perturb)
y_bounded = BoundedTensor(y_center, y_perturb)

# Compute bounds on squared norm
lb_sq, ub_sq = lirpa_model.compute_bounds(x=(x_bounded, y_bounded), method="backward")

# Take square root (monotone) to recover ||x-y||_2 bounds
lb, ub = lb_sq.clamp_min(0).sqrt().item(), ub_sq.clamp_min(0).sqrt().item()
print("Bounds for ||x-y||_2:", lb, ub)
