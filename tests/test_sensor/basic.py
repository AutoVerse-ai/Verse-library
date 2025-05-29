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
        car2 = CarAgent("car2", file_name=input_code_name)
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
    def test_set_states_2d_one_agent(self):
        disc = {}
        cont = {}
        state_dict = initialize_agents_examples("mono")
        agent = state_dict["car"]
        args = agent.decision_logic.args[0]
        arg_type = args.typ
        thing = args.name
        cont_var = agent.decision_logic.state_defs[arg_type].cont
        disc_var = agent.decision_logic.state_defs[arg_type].disc
        stat_var = agent.decision_logic.state_defs[arg_type].static
        val = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )
        print(cont_var, disc_var, stat_var)
        set_states_2d(disc, cont, thing, val, cont_var, disc_var, stat_var)
        print("disc is :", disc)
        print("cont is :", cont)
        self.assertTrue(True)
    def test_set_states_2d_multi_agent(self):
        disc = {}
        cont = {}
        state_dict = initialize_agents_examples("multi")

        agent_1 = state_dict["car1"]
        args_1 = agent_1.decision_logic.args[0]
        arg_type_1 = args_1.typ
        thing_1 = args_1.name
        cont_var_1 = agent_1.decision_logic.state_defs[arg_type_1].cont
        disc_var_1 = agent_1.decision_logic.state_defs[arg_type_1].disc
        stat_var_1 = agent_1.decision_logic.state_defs[arg_type_1].static
        val_1 = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )

        agent_2 = state_dict["car2"]
        args_2 = agent_2.decision_logic.args[0]
        arg_type_2 = args_2.typ
        thing_2 = args_2.name
        cont_var_2 = agent_2.decision_logic.state_defs[arg_type_2].cont
        disc_var_2 = agent_2.decision_logic.state_defs[arg_type_2].disc
        stat_var_2 = agent_2.decision_logic.state_defs[arg_type_2].static
        val_2 = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )
        set_states_2d(disc, cont, thing_1, val_1, cont_var_1, disc_var_1, stat_var_1)
        set_states_2d(disc, cont, thing_2, val_2, cont_var_2, disc_var_2, stat_var_2)

        print("disc is :", disc)
        print("cont is :", cont)

        self.assertTrue(True)

    def test_add_states_2d_one_agent(self):
        disc = {}
        cont = {}
        state_dict = initialize_agents_examples("mono")
        agent = state_dict["car"]
        args = agent.decision_logic.args[0]
        arg_type = args.typ
        thing = args.name
        cont_var = agent.decision_logic.state_defs[arg_type].cont
        disc_var = agent.decision_logic.state_defs[arg_type].disc
        stat_var = agent.decision_logic.state_defs[arg_type].static
        val = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )
        print(cont_var, disc_var, stat_var)
        add_states_2d(disc, cont, thing, val, cont_var, disc_var, stat_var)
        print("test_add_states_2d_one_agent disc is :", disc)
        print("test_add_states_2d_one_agent cont is :", cont)
        self.assertTrue(True)

    def test_add_states_2d_multi_agent(self):
        disc = {}
        cont = {}
        state_dict = initialize_agents_examples("multi")

        agent_1 = state_dict["car1"]
        args_1 = agent_1.decision_logic.args[0]
        arg_type_1 = args_1.typ
        thing_1 = args_1.name
        cont_var_1 = agent_1.decision_logic.state_defs[arg_type_1].cont
        disc_var_1 = agent_1.decision_logic.state_defs[arg_type_1].disc
        stat_var_1 = agent_1.decision_logic.state_defs[arg_type_1].static
        val_1 = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )

        agent_2 = state_dict["car2"]
        args_2 = agent_2.decision_logic.args[0]
        arg_type_2 = args_2.typ
        thing_2 = args_2.name
        cont_var_2 = agent_2.decision_logic.state_defs[arg_type_2].cont
        disc_var_2 = agent_2.decision_logic.state_defs[arg_type_2].disc
        stat_var_2 = agent_2.decision_logic.state_defs[arg_type_2].static
        val_2 = (
            [0.0, 1.0, 2.0, 3.0],
            ["up", "down", "left", "right"],
            ["a", "b", "c", "d"]
        )
        add_states_2d(disc, cont, thing_1, val_1, cont_var_1, disc_var_1, stat_var_1)
        add_states_2d(disc, cont, thing_2, val_2, cont_var_2, disc_var_2, stat_var_2)

        print("disc is :", disc)
        print("cont is :", cont)

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
    print("Test sensor complete")

