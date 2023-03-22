import functools, pprint, random, math
from verse.agents.example_agent import CarAgent
from verse.analysis.utils import wrap_to_pi
from verse.map.example_map.intersection import Intersection
from verse.scenario.scenario import Benchmark
pp = functools.partial(pprint.pprint, compact=True, width=130)

from controller.intersection_car import AgentMode

CAR_NUM = 7
CAR_ACCEL_RANGE = (0.7, 3)
CAR_SPEED_RANGE = (1, 3)
CAR_THETA_RANGE = (-0.1, 0.1)

def rand(start: float, end: float) -> float:
    return random.random() * (end - start) + start

def run(meas=False):
    traces = bench.run(60, 0.05)

    if bench.config.dump:
        traces.dump_tree()
        traces.dump("main.json") 
        traces.dump("tree2.json" if meas else "tree1.json") 

    if bench.config.plot and meas:
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
    if len(sys.argv) == 3:
        seed = int(sys.argv[2])
    else:
        seed = int(time.time())
    random.seed(seed)

    dirs = "WSEN"
    map = Intersection()
    bench.scenario.set_map(map)
    for i in range(CAR_NUM):
        car = CarAgent(f"car{i}", file_name=ctlr_src, speed=rand(*CAR_SPEED_RANGE), accel=rand(*CAR_ACCEL_RANGE))
        # if i == 0:
        #     for p in car.decision_logic.paths:
        #         print(ast_dump(p.val_veri), "<=", ast_dump(p.cond_veri))
        bench.scenario.add_agent(car)
        dir = random.randint(0, 3)
        src = dirs[dir]
        dst_dirs = list(dirs)
        dst_dirs.remove(src)
        dst = dst_dirs[random.randint(0, 2)]
        lane = random.randint(0, map.lanes - 1)
        start, off = map.size + rand(0, map.length * 0.3), rand(0, map.width) + map.width * lane
        pos = { "N": (-off, start), "S": (off, -start), "W": (-start, -off), "E": (start, off) }[src]
        init = [*pos, wrap_to_pi(dir * math.pi / 2 + rand(*CAR_THETA_RANGE)), rand(*CAR_SPEED_RANGE)]
        modes = (AgentMode.Accel, f"{src}{dst}_{lane}")
        bench.scenario.set_init_single(car.id, (init,), modes)

    if 'b' in bench.config.args:
        run(True)
    elif 'r' in bench.config.args:
        run()
        run(True)
    print("seed:", seed)
