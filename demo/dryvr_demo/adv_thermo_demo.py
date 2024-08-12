from origin_agent import adv_thermo_agent
from verse import Scenario
from verse.analysis.analysis_tree import first_transitions
from verse.parser.parser import ControllerIR
from verse.plotter.plotter2D import *
from verse.scenario.scenario import Benchmark, ScenarioConfig
from verse.sensor.example_sensor.thermo_sensor import ThermoSensor
import plotly.graph_objects as go
from enum import Enum, auto
import sys, time


class ThermoMode(Enum):
    WARM = auto()
    WARM_FAST = auto()
    COOL = auto()
    COOL_FAST = auto()


def run(meas=False):
    if bench.config.sim:
        bench.scenario.simulator.cache_hits = (0, 0)
    else:
        bench.scenario.verifier.tube_cache_hits = (0, 0)
        bench.scenario.verifier.trans_cache_hits = (0, 0)
    if not meas and not bench.scenario.config.incremental:
        return
    traces = bench.run(RUN_TIME, TIME_STEP)

    if bench.config.dump:
        traces.dump("tree2.json" if meas else "tree1.json")

    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(traces, None, fig, 2, 1, [2, 1], "lines", "trace")
        fig.show()

    if meas:
        bench.report()
    print(f"agent transition times: {first_transitions(traces)}")


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/adv_thermo_controller.py"

    bench = Benchmark(sys.argv)
    bench = Benchmark(sys.argv)

    bench.scenario.add_agent(adv_thermo_agent("test", file_name=input_code_name))
    bench.scenario.add_agent(adv_thermo_agent("test2", file_name=input_code_name))
    bench.scenario.set_sensor(ThermoSensor())
    # modify mode list input
    bench.scenario.set_init(
        [
            [[75.0, 0.0, 0.0], [75.0, 0.0, 0.0]],
            [[76.0, 0.0, 0.0], [76.0, 0.0, 0.0]],
        ],
        [
            tuple([ThermoMode.WARM]),
            tuple([ThermoMode.COOL]),
        ],
    )
    if "b" in bench.config.args:
        run(True)
    elif "r" in bench.config.args:
        run()
        run(True)
    elif "n" in bench.config.args:
        run()
        bench.scenario.set_init_single(
            "test2", [[77.0, 0.0, 0.0], [77.0, 0.0, 0.0]], tuple([ThermoMode.WARM])
        )
        run(True)
    elif "1" in bench.config.args:
        run()
        bench.swap_dl("test", input_code_name.replace(".py", "2.py"))
        run(True)
    elif "2" in bench.config.args:
        run()
        bench.swap_dl("test2", input_code_name.replace(".py", "2.py"))
        run(True)
    if bench.scenario.config.parallel:
        import ray, time

        ray.timeline(f"adv_thermo_{int(time.time())}.json")
