import functools, pprint, random, math
from typing import Optional, Tuple
from verse.agents.example_agent import CarAgentDebounced
from verse.analysis.utils import wrap_to_pi
from verse.map.example_map.intersection import Intersection
from verse.scenario.scenario import Benchmark
pp = functools.partial(pprint.pprint, compact=True, width=130)

from controller.intersection_car import AgentMode

CAR_NUM = 5
LANES = 3
CAR_ACCEL_RANGE = (0.7, 3)
CAR_SPEED_RANGE = (1, 3)
CAR_THETA_RANGE = (-0.1, 0.1)

def rand(start: float, end: float) -> float:
    return random.random() * (end - start) + start

def run(meas=False):
    if bench.config.sim:
        bench.scenario.simulator.cache_hits = (0, 0)
    else:
        bench.scenario.verifier.tube_cache_hits = (0, 0)
        bench.scenario.verifier.trans_cache_hits = (0, 0)
    traces = bench.run(30, 0.05)

    if bench.config.dump:
        traces.dump("tree2.json" if meas else "tree1.json") 

    if bench.config.plot:
        import plotly.graph_objects as go
        from verse.plotter.plotter2D import reachtube_tree, simulation_tree

        fig = go.Figure()
        if bench.config.sim:
            fig = simulation_tree(traces, bench.scenario.map, fig, 1, 2, print_dim_list=[1, 2])
        else:
            fig = reachtube_tree(traces, bench.scenario.map, fig, 1, 2, [1, 2], 'lines',combine_rect=5)
        fig.show()

    if meas:
        bench.report()

if __name__ == "__main__":
    import sys
    bench = Benchmark(sys.argv)
    ctlr_src = "demo/vehicle/controller/intersection_car.py"
    import time
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    else:
        seed = int(time.time())

    if len(sys.argv) == 5:
        CAR_NUM = int(sys.argv[3])
        LANES = int(sys.argv[4])

    print()
    print("------------------------  ", sys.argv[1], "   ==============")
    print("seed: %d, CAR_NUM %d, LANES: %d" %(seed, CAR_NUM, LANES))
    #print("seed:", seed)
    random.seed(seed)

    dirs = "WSEN"
    map = Intersection(lanes=LANES, length=400)
    bench.scenario.set_map(map)
    def set_init(id: str, alt_pos: Optional[Tuple[float, float]] = None):
        dir = random.randint(0, 3)
        src = dirs[dir]
        dst_dirs = list(dirs)
        dst_dirs.remove(src)
        dst = dst_dirs[random.randint(0, 2)]
        mid_lane_ind = int(map.lanes / 2 - 1)
        lane = random.randint(mid_lane_ind, mid_lane_ind + 1)
        start, off = (map.size + rand(0, map.length * 0.3), rand(0, map.width) + map.width * lane)
        pos = { "N": (-off, start), "S": (off, -start), "W": (-start, -off), "E": (start, off) }[src] if alt_pos == None else alt_pos
        init = [*pos, *((wrap_to_pi(dir * math.pi / 2 + rand(*CAR_THETA_RANGE)), rand(*CAR_SPEED_RANGE)) if alt_pos == None else bench.scenario.init_dict[id][0][2:4]), 0]
        assert len(init) == 5, bench.scenario.init_dict
        modes = (AgentMode.Accel, f"{src}{dst}_{lane}") if alt_pos == None else bench.scenario.init_mode_dict[id]
        bench.scenario.set_init_single(id, (init,), modes)
    car_id = lambda i: f"car{i}"

    for i in range(CAR_NUM):
        car = CarAgentDebounced(car_id(i), file_name=ctlr_src, speed=rand(*CAR_SPEED_RANGE), accel=rand(*CAR_ACCEL_RANGE))
        bench.scenario.add_agent(car)
        set_init(car.id)

    if 'b' in bench.config.args:
        run(True)
    elif 'r' in bench.config.args:
        run()
        run(True)
    elif 'n' in bench.config.args:
        run()
        set_init(car_id(3), (100, -0.8))
        run(True)
    elif '3' in bench.config.args:
        run()
        old_agent = bench.scenario.agent_dict["car3"]
        bench.scenario.agent_dict["car3"] = CarAgentDebounced('car3', file_name=ctlr_src.replace(".py", "_sw5.py"),
                                                              speed=old_agent.speed, accel=old_agent.accel)
        run(True)
    print("seed:", seed)
