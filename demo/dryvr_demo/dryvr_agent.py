# Example agent.
from re import L
from typing import Tuple, List

import numpy as np
from scipy.integrate import odeint

from verse.agents import BaseAgent
from verse.map import LaneMap
from verse.parser import ControllerIR

class LaubLoomisAgent(BaseAgent):
    def __init__(self, id, code = None, file_name = None):
        self.decision_logic: ControllerIR = ControllerIR.empty()
        self.id = id
        self.init_cont = None 
        self.init_disc = None
        self.static_parameters = None 
        self.uncertain_parameters = None

    def dynamics(self, t, x):
        x1, x2, x3,\
        x4, x5, x6,\
        x7 = x 

        dx1 = 1.4*x3-0.9*x1 
        dx2 = 2.5*x5-1.5*x2 
        dx3 = 0.6*x7-0.8*x2*x3 
        dx4 = 2-1.3*x3*x4 
        dx5 = 0.7*x1-x4*x5 
        dx6 = 0.3*x1-3.1*x6 
        dx7 = 1.8*x6-1.5*x2*x7  

        return [dx1,dx2,dx3,dx4,dx5,dx6,dx7]

class QuadrotorAgent(BaseAgent):
    def __init__(self, id, code = None, file_name = None):
        self.decision_logic: ControllerIR = ControllerIR.empty()
        self.id = id
        self.init_cont = None 
        self.init_disc = None
        self.static_parameters = None 
        self.uncertain_parameters = None

    def dynamics(self, t, x):
        x1, x2, x3,\
        x4, x5, x6,\
        x7, x8, x9,\
        x10, x11, x12, u1 = x 

        u2 = 0
        u3 = 0

        g = 9.81
        R = 0.1 
        l = 0.5
        Mrotor = 0.1
        M = 1 
        m = M+4*Mrotor 

        Jx = 2/5*M*R**2+2*l**2*Mrotor 
        Jy = Jx 
        Jz = 2/5*M*R**2+4*l**2*Mrotor 

        F = m*g-10*(x3-u1)+3*x6 
        tau_phi = -(x7-u2)-x10 
        tau_theta = -(x8-u3)-x11 
        tau_psi  = 0 

        dx1 = np.cos(x8)*np.cos(x9)*x4+(np.sin(x7)*np.sin(x8)*np.cos(x9)-np.cos(x7)*np.sin(x9))*x5+(np.cos(x7)*np.sin(x8)*np.cos(x9)+np.sin(x7)*np.sin(x9))*x6 
        dx2 = np.cos(x8)*np.sin(x9)*x4+(np.sin(x7)*np.sin(x8)*np.sin(x9)+np.cos(x7)*np.cos(x9))*x5+(np.cos(x7)*np.sin(x8)*np.sin(x9)-np.sin(x7)*np.cos(x9))*x6  
        dx3 = np.sin(x8)*x4-np.sin(x7)*np.cos(x8)*x5-np.cos(x7)*np.cos(x8)*x6 
        dx4 = x12*x5-x11*x6-g*np.sin(x8)
        dx5 = x10*x6-x12*x4+g*np.cos(x8)*np.sin(x7)
        dx6 = x11*x4-x10*x5+g*np.cos(x8)*np.cos(x7)-F/m 
        dx7 = x10+np.sin(x7)*np.tan(x8)*x11+np.cos(x7)*np.tan(x8)*x12
        dx8 = np.cos(x7)*x11-np.sin(x7)*x12 
        dx9 = np.sin(x7)/np.cos(x8)*x11+np.cos(x7)/np.cos(x8)*x12 
        dx10 = (Jy-Jz)/Jx*x11*x12+1/Jx*tau_phi
        dx11 = (Jz-Jx)/Jy*x10*x12+1/Jy*tau_theta 
        dx12 = (Jx-Jy)/Jz*x10*x11+1/Jz*tau_psi
        du1 = 0 

        return [dx1,dx2,dx3,dx4,dx5,dx6,dx7,dx8,dx9,dx10,dx11,dx12,du1]
