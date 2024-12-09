# Example agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
from scipy.integrate import solve_ivp

from verse.agents import BaseAgent
from verse.map import LaneMap
import os 
import pandas as pd 
from scipy.spatial.transform import Rotation
from verse.parser import ControllerIR

script_dir = os.path.dirname(os.path.realpath(__file__))

class MiniHawkAgent(BaseAgent):
    def __init__(self, id, code=None, file_name=None, folder_name = './MiniHawk_Obstacles-None_Landing-Rooftop_Uncertainty-Uniform-5m'):
        # Calling the constructor of tha base class

        # super().__init__(id, code, file_name)
        self.id = id
        output_dir = folder_name
        self.traces_list = []
        for i, name in enumerate(os.listdir(output_dir)):
            # if i==1:
            #     continue
            if name.startswith('extracted'):
                df = pd.read_csv(os.path.join(output_dir, name, './_minihawk_pose.csv'))
                self.traces_list.append(df)
        self.decision_logic: ControllerIR = ControllerIR.empty()
        self.process_traces()
        self.internal_counter = 0
    
    def process_traces(self):
        min_trace_length = np.inf
        for trace_df in self.traces_list:
            timestamp = np.array(trace_df['timestamp'])
            if len(timestamp) < min_trace_length:
                min_trace_length = len(timestamp)

        self.all_traces = []
        for trace_df in self.traces_list:
            t = np.array(trace_df['timestamp'])[:min_trace_length:2]
            tx = np.array(trace_df['tx'])[:min_trace_length:2]
            ty = np.array(trace_df['ty'])[:min_trace_length:2]
            tz = np.array(trace_df['tz'])[:min_trace_length:2]
            rx = np.array(trace_df['rx'])[:min_trace_length:2]
            ry = np.array(trace_df['ry'])[:min_trace_length:2]
            rz = np.array(trace_df['rz'])[:min_trace_length:2]
            rw = np.array(trace_df['rw'])[:min_trace_length:2]

            # quats = np.zeros((len(rx),4))
            # quats[:,0],quats[:,1],quats[:,2],quats[:,3] = rx,ry,rz,rw 
            # rpy = Rotation.from_quat(quats).as_euler('xyz')
            t = np.round(t-t[0],2)
            t_multiplier = np.array([i for i in range(len(t))])
            t = np.round(t_multiplier*0.25, 4)
            quats = np.zeros((len(t), 4))
            quats[:,0] = rx 
            quats[:,1] = ry 
            quats[:,2] = rz 
            quats[:,3] = rw
            rpy = Rotation.from_quat(quats).as_euler('xyz')
            
            trace = np.zeros((len(t), 7))
            trace[:,0] = t 
            trace[:,1] = tx
            trace[:,2] = ty
            trace[:,3] = tz
            trace[:,4:] = rpy 
            self.all_traces.append(trace)
        self.all_traces = np.array(self.all_traces)

    def generate_nominal_trace(self):
        self.nominal_trace = np.mean(self.all_traces, axis=0)

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map: LaneMap = None, idx = 0) -> np.ndarray:
        '''
        # time_bound = float(time_bound)
        # number_points = int(np.ceil(time_bound/time_step))
        # t = [round(i*time_step, 10) for i in range(0, number_points)]

        # init = initialCondition
        # trace = [[0]+init]
        # for i in range(len(t)):
        #     r = self.action_handler(mode[0])
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten().tolist()
        #     trace.append([t[i] + time_step] + init)
        # return np.array(trace)
        '''
        # pass   
        steps = int(time_bound/time_step)
        if idx == 0:
            return self.nominal_trace[:steps,:4]
        else:
        # if self.internal_counter == 0:
        #     initial_condition = np.array(initialCondition)      
        #     all_inits = self.all_traces[:,0,1:]
        #     dists = np.linalg.norm(all_inits - initial_condition, axis=1)
        #     trace_idx = np.argmin(dists)
        #     init = initialCondition
        #     print(trace_idx, initialCondition, dists)
        #     self.nominal_idx = trace_idx
        #     self.internal_counter += 1
        # else:
            if idx >= self.all_traces.shape[0]:
                idx = idx%self.all_traces.shape[0]
            trace_idx = idx
            # self.internal_counter += 1

        # time_bound = float(time_bound)
        # num_points = int(np.ceil(time_bound / time_step))
        # trace = np.zeros((num_points + 1, 1 + len(init)))
        # trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        # trace[0, 1:] = init

        # for i in range(num_points):
        #     r = self.action_handler(mode[0])
        #     r.set_initial_value(init)
        #     res: np.ndarray = r.integrate(r.t + time_step)
        #     init = res.flatten()
        #     # if init[3] < 0:
        #     #     init[3] = 0
        #     trace[i + 1, 0] = time_step * (i + 1)
        #     trace[i + 1, 1:] = init
        return self.all_traces[trace_idx,:steps,:4]

if __name__ == "__main__":
    agent = MiniHawkAgent('a')