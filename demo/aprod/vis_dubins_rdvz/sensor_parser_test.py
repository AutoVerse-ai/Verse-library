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

from sensor_parser_hybrid import parsed_sensor
from prox_error_all_bounds import angular_span_between_rects, angular_bounds_diff, angular_span_rect, angular_bounds_diff_correct, combine_angular_bounds


def bounded_subtract(a, b):
    """
    Bounding a - b, returns [(a-b)_min, (a-b)_max]
    
    :param a: float
    :param b: float
    """
    a_min, a_max = a
    b_min, b_max = b
    return [a_min - b_max, a_max - b_min]

def atan2_piecewise(x, y):
    """
    atan2 implemented as a piecewise function
    not sure if I'm going to use this but have it as an option
    
    :param x: float
    :param y: float
    """
    # include offset so I don't have to worry about boundary cases at axes
    if x>=0:
        return np.arctan(y/(x+1e-8))
    if x<0 and y>=0:
        return np.arctan(y/(x-1e-8)) + np.pi
    else: # x<0 and y<0
        return np.arctan(y/(x-1e-8)) - np.pi
    
def vis_sensor(x, y, theta, x_other, y_other, phi):
    rel_x, rel_y = x_other-x, y_other-y
    eta = np.arctan2(rel_y, rel_x) # rewrite/redo this as a different function
    psi = (eta-theta+np.pi) % (2*np.pi) - np.pi # maybe delinate as wrap_to_pi/special modular arithematic function
    if psi >= 0 and psi <= phi:
        return -1
    if psi < 0 and psi >= - phi:
        return 1
    if psi > phi:
        return -2
    else: # psi < -phi
        return 2

# def bounded_vis_sensor(x, y, theta, x_other, y_other, phi):
def bounded_vis_sensor(x, y, theta, x_other, y_other):
    """
    Manually bounding the vis_sensor function
    
    :param x: x-coord of ego
    :param y: y-coord of ego
    :param theta: theta/heading of ego
    :param x_other: x-coord of other
    :param y_other: y-coord of other
    :param phi: (half) angular range of sensor
    """
    def rel(x,y,x_other, y_other):
        rel_x, rel_y = x_other-x, y_other-y
        return rel_x, rel_y
    phi = 1
    rel_x, rel_y = bounded_subtract(x_other,x), bounded_subtract(y_other, y)
    # rel_x, rel_y = parsed_sensor(rel, input_bounds=np.array([x, y, x_other, y_other]), num_splits=1) # these two do the same, just verifying that instantiating new functions work
    eta = angular_span_rect(rel_x + rel_y, split=True) # these are angular bounds s.t. eta_max can be < eta_min if there is wrapping about the discountinty -- should this be like this or should I just split the bound up front?
    # eta can be a tuple of tuples if the rel rect crosses the discontinuity
    psi_bounds = []
    for eta_bound in eta:
        psi_bounds.append(angular_bounds_diff_correct(eta_bound, theta))
    # TODO: combine angular bounds as best as possible
    psi_bounds = combine_angular_bounds(psi_bounds)

    outputs = set()
    for psi_bound in psi_bounds:
        psi_min, psi_max = psi_bound
        
        if not (psi_max < 0 or psi_min > phi): # TODO: find a more generic way to do this -- should be able to abstract entire if chain to this
            outputs.add(-1)
        if not (psi_max <= -phi or psi_min >= 0):
            outputs.add(1)
        if psi_max > phi:
            outputs.add(-2)
        if psi_min < -phi:
            outputs.add(2)

    return min(outputs), max(outputs)
    # return psi_bounds
    # return eta

if __name__ == "__main__":
    # input = [0, 0, 0, 1, 1, 1]
    # out = vis_sensor(*input)
    # print(out)

    # x, y, theta, x_other, y_other, phi = [0.1,0.1], [0.1,0.1], [0,0], [0.9, 1.1], [0.9, 1.1], [1, 1]
    x, y, theta, x_other, y_other = [0,0], [0.1,0.1], [0,0], [0.9, 1.1], [-1.1, 1.5]
    input_bounds = [x, y, theta, x_other, y_other]
    out_bounds = bounded_vis_sensor(*input_bounds)
    print(out_bounds)
    exit()