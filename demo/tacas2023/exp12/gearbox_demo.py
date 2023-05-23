from sleeve_agent import sleeve_agent
from verse.analysis.analysis_tree import first_transitions
from verse.scenario.scenario import Benchmark
from verse.plotter.plotter2D import *
import plotly.graph_objects as go
from enum import Enum, auto
import sys
import time


class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


sim_length, time_step = 1, 0.0001


def run(meas=False):
    if bench.config.sim:
        bench.scenario.simulator.cache_hits = (0, 0)
    else:
        bench.scenario.verifier.tube_cache_hits = (0, 0)
        bench.scenario.verifier.trans_cache_hits = (0, 0)
    if not meas and not bench.scenario.config.incremental:
        return
    traces = bench.run(sim_length, time_step)

    if bench.config.dump:
        traces.dump("tree2.json" if meas else "tree1.json")

    if bench.config.plot:
        fig = go.Figure()
        fig = reachtube_tree(
            traces, None, fig, 1, 2, [1, 2, 3, 4, 5], "lines", "trace", sample_rate=1
        )
        fig.show()

    if meas:
        bench.report()
    print(f"agent transition times: {first_transitions(traces)}")


if __name__ == "__main__":
    input_code_name = "./demo/tacas2023/exp12/sleeve_controller.py"
    bench = Benchmark(sys.argv, init_seg_length=1)
    bench.agent_type = "G"
    bench.noisy_s = "N/A"

    bench.scenario.add_agent(sleeve_agent("sleeve", file_name=input_code_name))
    bench.scenario.set_init(
        [[[-0.0168, 0.0029, 0, 0, 0], [-0.0166, 0.0031, 0, 0, 0]]], [tuple([AgentMode.Free])]
    )
    if "b" in bench.config.args:
        run(True)
    elif "r" in bench.config.args:
        run()
        run(True)
    elif "n" in bench.config.args:
        run()
        bench.scenario.set_init(
            [[[-0.0168, 0.0030, 0, 0, 0], [-0.0166, 0.0032, 0, 0, 0]]], [tuple([AgentMode.Free])]
        )
        run(True)
    elif "1" in bench.config.args:
        run()
        bench.swap_dl("sleeve", input_code_name.replace(".py", "2.py"))
        run(True)
    bench.report()
