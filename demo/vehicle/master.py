import functools, pprint, random, math
import ray
from verse.agents.example_agent import CarAgentDebounced
from verse.analysis.utils import wrap_to_pi
from verse.map.example_map.intersection import Intersection
from verse.scenario.scenario import Benchmark

pp = functools.partial(pprint.pprint, compact=True, width=130)

from controller.intersection_car import AgentMode
import datetime
import subprocess
import os, sys

if __name__ == "__main__":

    # output = sys.stdout
    LANES_LIST = [4]
    CAR_NUM_LIST = [9, 10, 12,15]
    RUN_TIME_LIST = [15, 30, 40, 50]
    RUN_MODE_LIST = ['blv']
    RANDOM_SEED = 1118    #460, 1118, 1682538796
    alt_car_list = []
    OUTPUT_FILENAME = "output-v-8cores(1118)(0502)"
    
    #First check how many CPU is online
    with open(OUTPUT_FILENAME + ".txt", "a") as f:
        print(file=f)
        print("============================", file=f)
        
    output=open(OUTPUT_FILENAME + ".txt", "a")
    subprocess.call(["grep", "processor", "/proc/cpuinfo"], stdout=output, stderr=output)

    my_env = os.environ.copy()
    my_env["RAY_PROFILING"] = "1"
    for LANE in LANES_LIST:
        for CAR_NUM in CAR_NUM_LIST:
            for RUN_TIME in RUN_TIME_LIST:
                for run_mode in RUN_MODE_LIST:
                    print(datetime.datetime.now(), file=open(OUTPUT_FILENAME + ".txt", "a"))
                    subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection.py", run_mode, f"SEED={RANDOM_SEED}", f"CAR_NUM={CAR_NUM}", f"LANES={LANE}", f"RUN_TIME={RUN_TIME}", f"OUTPUT={OUTPUT_FILENAME}"], 
                                    stdout=output, stderr=output, env=my_env)

    print(datetime.datetime.now(), file=open(OUTPUT_FILENAME + ".txt", "a"))
    
    # verify
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "b", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # # verify_parallel
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "bl", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # verify_incrmental
    # x_list = [77.09140887861837, 80, 85, 100, 120, 150]
    # y_list = [9, 10, 10.5, 11, 11.619671922657663]
    #start:106.07500751049828, off: 2.5459862275151646
    # x_list = [100]
    # y_list = [3]
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i in [0, 2, 4, 6]:
    #     alt_car_list = [i]
    #     subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [1, 3 ]:
    #     for i2 in [5, 7]:
    #         alt_car_list = [i1, i2]
    #         subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [0]:
    #     for i2 in [2, 4]:
    #         for i3 in [6, 8]:
    #             alt_car_list = [i1, i2, i3]
    #             subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [1, 3]:
    #     for i2 in [4, 6]:
    #         for i3 in [7]:
    #             for i4 in [8]:
    #                 alt_car_list = [i1, i2, i3, i4]
    #                 subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [0]:
    #     for i2 in [2]:
    #         for i3 in [4]:
    #             for i4 in [5, 6]:
    #                 for i5 in [7, 8]:
    #                     alt_car_list = [i1, i2, i3, i4, i5]
    #                     subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [0]:
    #     for i2 in [1]:
    #         for i3 in [2, 3]:
    #             for i4 in [4, 5]:
    #                 for i5 in [7]:
    #                     for i6 in [8]:
    #                         alt_car_list = [i1, i2, i3, i4, i5, i6]
    #                         subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [0, 1]:
    #     for i2 in [2]:
    #         for i3 in [3]:
    #             for i4 in [4]:
    #                 for i5 in [5]:
    #                     for i6 in [6]:
    #                         for i7 in [7, 8]:
    #                             alt_car_list = [i1, i2, i3, i4, i5, i6, i7]
    #                             subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # for i1 in [0, 1]:
    #     for i2 in [2]:
    #         for i3 in [3]:
    #             for i4 in [4]:
    #                 for i5 in [5]:
    #                     for i6 in [6]:
    #                         for i7 in [7]:
    #                             for i8 in [8]:
    #                                 alt_car_list = [i1, i2, i3, i4, i5, i6, i7, i8]
    #                                 subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)

    # alt_car_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "in", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(alt_car_list) ], stdout=output, stderr=output)
    # # for x in x_list:
    # #     for y in y_list:
    # #         subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "inv", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(x), str(y) ], stdout=output, stderr=output)
    # print(time.time)

## Large Nodes

    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "b", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(x), str(y) ], stdout=output, stderr=output)
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "bl", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(x), str(y) ], stdout=output, stderr=output)
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "bv", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(x), str(y) ], stdout=output, stderr=output)
    # subprocess.call(["/usr/bin/time", "-v", "python", "demo/vehicle/intersection_inc.py", "bvl", str(RANDOM_SEED), str(CAR_NUM), str(LANES), str(x), str(y) ], stdout=output, stderr=output)

    # output.close()