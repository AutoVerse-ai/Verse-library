# this file contains helper functions for mp

import numpy as np
from typing import Any, Tuple, Dict, Callable, List

def generate_init(init_set: List[List[float]]):
    """
    TODO:   given the initial set with the upper and lower bound, 
            generate multiple initial points located in the initial set
            as the input of multiple simulation.
    """
    ############## Your Code Start Here ##############
    init_point_list = []
    num_sample = 10

    np.random.seed(2023)
    for i in range(num_sample):
        res = np.random.uniform(init_set[0], init_set[1]).tolist()
        init_point_list.append(res)
    ############## Your Code End Here ##############
    return init_point_list

def simulate_multi(init_points):
    """
    given a list of multiple initial points, 
    run simulation, evaluate avg velocity and check safety for every point. 
    you may also get visualization result conmbined in one plot.
    For performance concern, do not use too many initial points when enabling visualization.
    """
    pass

if __name__ == "__main__":
    init_point_list = generate_init([[0,0,0,10,0],[2,1,0,10,0]])
    print(init_point_list)