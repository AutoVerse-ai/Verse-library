from parser_wrapper import parse_function
import numpy as np
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
        return -2 # parser should be able to parse out arbitrary functions
    else:
        return 2

def pw(psi, phi):
    y = vis_sensor_piecewise(psi, phi)
    return y

def vis_sensor(x,y,x_other,y_other,psi, phi, theta):
    rel_x = x_other - x
    rel_y = y_other - y
    eta = atan2(rel_x, rel_y) # FIXME: bounded_arctan2 has wrong form, this is equivalent to atan2(y,x)
    psi = eta-theta
    obs = vis_sensor_piecewise(psi, phi)
    return obs

# TODO: add a way to automatically create the input bounds given a list of bounds and the function header -- user shouldn't need to do this by hand
# input_bounds = {'x': (0, 1), 'y': (1, 2)}
# input_bounds = {'x': [(0, 1)], 'y': [(1, 2), (2,3)]}
# input_bounds = {'x': [(-1, -0.5)], 'y': [(-1, 0.5)]} 
# final_bounds, history = parse_function(test_sensor, input_bounds)


# input_bounds = {'x': [(134, 134.01)], 'y': [(11.5, 11.51)]} # TODO: check to see if this is consistent with the stanley scenario
# script_dir = os.path.realpath(os.path.dirname(__file__))
# tmp_map = opendrive_map(os.path.join(script_dir, "t1_triple.xodr"))
# final_bounds, history = parse_function(map_sensor, input_bounds, track_map=tmp_map, track_mode=TrackMode.T1)

# input_bounds = {'psi': [(-0.9, 2)], 'phi': [(1,1)]}
# final_bounds, history = parse_function(pw, input_bounds, [vis_sensor_piecewise]) # NOTE: is this the best way to handle piecewise functions? it's easy, but slightly unintuitive

# TODO: create function that automatically converts list of bounds into a dict like below
input_bounds = {'x':[(0,0)],'y':[(0.1,0.1)],'theta':[(0.1,0.1)],
                'x_other':[(0.9,1.1)],'y_other':[(-1.1,1.5)], 'phi':[(1,1)]}

final_bounds, history = parse_function(vis_sensor, input_bounds, [vis_sensor_piecewise])

print("\nFinal bounds:")
for var, bounds in final_bounds.items():
    for bound in bounds:
        print(f"  {var} ∈ [{bound[0]:.6f}, {bound[-1]:.6f}]")