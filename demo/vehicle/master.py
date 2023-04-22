import functools, pprint, random, math

from verse.agents.example_agent import CarAgentDebounced

from verse.analysis.utils import wrap_to_pi

from verse.map.example_map.intersection import Intersection

from verse.scenario.scenario import Benchmark

pp = functools.partial(pprint.pprint, compact=True, width=130)



from controller.intersection_car import AgentMode


import time
CAR_NUM_LIST = [10]

LANES_LIST = [3]

RANDOM_SEED_LIST = range(5,101)

if __name__ == "__main__":

    import subprocess
    import sys

    output=open("random-output.txt", "a")
    #output = sys.stdout
    for CAR_NUM in CAR_NUM_LIST:
        for LANES in LANES_LIST:
            for RANDOM_SEED in RANDOM_SEED_LIST:
                subprocess.call(["python", "demo/vehicle/intersection.py", "b", str(RANDOM_SEED), str(CAR_NUM), str(LANES) ], stdout=output, stderr=output)
                #subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection.py", "b", str(RANDOM_SEED), str(CAR_NUM), str(LANES) ], stdout=output, stderr=output)
            #subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection.py", "bpl", "4", str(CAR_NUM), str(LANES) ], stdout=output, stderr=output)

         


