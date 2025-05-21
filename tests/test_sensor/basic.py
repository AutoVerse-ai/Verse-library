from typing import Dict
from verse.sensor import BaseSensor
from verse.agents import BaseAgent
from verse.sensor.base_sensor import sets, set_states_2d, set_states_3d, add_states_2d, add_states_3d, adds
from verse.agents.example_agent import NPCAgent, CarAgent
import unittest
import os

def initialize_agents_examples(mode : str) -> Dict[str, BaseAgent]:
    assert mode in ["mono", "multi"]
    if mode == "mono":
        # Only one agent is generated
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car = CarAgent("car", file_name=input_code_name)
        return {"car": car}
    if mode == "multi":
        # Multi-agents is generated
        script_dir = os.path.realpath(os.path.dirname(__file__))
        input_code_name = os.path.join(script_dir, "../test_controller/example_controller5.py")
        car1 = CarAgent("car1", file_name=input_code_name)
        car2 = NPCAgent("car2")
        return {"car1":car1, "car2":car2}
    return

class TestSensorBasic(unittest.TestCase):
    def test_sets(self):
        d = {}
        sets(d, "car", ["color", "speed"], ["red", 100])
        self.assertTrue(len(d) == 2)
        self.assertTrue(list(d.keys()) == ["car.color", "car.speed"])
        self.assertTrue(list(d.values()) == ["red", 100])
    def test_add_1(self):
        d = {}
        adds(d, "car", ["color", "speed"], ["red", 100])
        self.assertTrue(len(d) == 2)
        self.assertTrue(list(d.keys()) == ["car.color", "car.speed"])
        self.assertTrue(list(d.values()) == [["red"], [100]])
    def test_add_2(self):
        d = {}
        adds(d, "car", ["color", "speed"], ["red", 100])
        adds(d, "car", ["speed"], [120])
        self.assertTrue(list(d.values()) == [["red"], [100, 120]])
    
    # NOTE: This test case returns errors
    # Here is the log of the error
    #   Traceback (most recent call last):
    #   File "/Users/bachhoang/Verse-library/tests/test_sensor/basic.py", line 55, in test_set_states_2d_one_agent
    #     set_states_2d(disc, cont, thing, agent, cont_var, disc_var, stat_var)
    #   File "/Users/bachhoang/miniconda3/envs/verse_env/lib/python3.11/site-packages/verse/sensor/base_sensor.py", line 18, in set_states_2d
    #     state, mode, static = val
    #     ^^^^^^^^^^^^^^^^^^^
    # TypeError: cannot unpack non-iterable CarAgent object
    # def test_set_states_2d_one_agent(self):
    #     disc = {}
    #     cont = {}
    #     state_dict = initialize_agents_examples("mono")
    #     agent = state_dict["car"]
    #     args = agent.decision_logic.args[0]
    #     arg_type = args.typ
    #     thing = args.name
    #     cont_var = agent.decision_logic.state_defs[arg_type].cont
    #     disc_var = agent.decision_logic.state_defs[arg_type].disc
    #     stat_var = agent.decision_logic.state_defs[arg_type].static
    #     set_states_2d(disc, cont, thing, agent, cont_var, disc_var, stat_var)
    #     print("disc is :", disc)
    #     print("cont is :", cont)
    #     self.assertTrue(True)
    # def test_set_states_2d_multi_agent(self):
    #     disc = {}
    #     cont = {}
    #     state_dict = initialize_agents_examples("multi")
    #     self.assertTrue(True)




if __name__ == "__main__":
    unittest.main()
    print("Test sensor complete")

