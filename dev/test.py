import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt 

def dynamics(t,state,u):
    x1, x2, x3, x1_hat, x2_hat, x3_hat = state
    w1, w1_hat = -0.1, 0.1
