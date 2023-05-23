# Introducing unittests for DryVR++ development process
# Read more from https://docs.python.org/3/library/unittest.html

# A scenario is created for testing
from enum import Enum, auto
import unittest


# Defines the set of possible driving modes of a vehicle
class VehicleMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    Lane0 = auto()


# Defines continuous states
class State:
    x = 0.0
    y = 0.0
    theta = 0.0
    v = 0.0
    vehicle_mode: VehicleMode = VehicleMode.Normal
    lane_mode: TrackMode = TrackMode.Lane0

    def __init__(self, x, y, theta, v, vehicle_mode: VehicleMode, lane_mode: TrackMode):
        self.data = []


def controller(ego: State):
    output_vehicle_mode = ego.vehicle_mode
    output_lane_mode = ego.lane_mode
    if ego.vehicle_mode == VehicleMode.Normal:
        if ego.lane_mode == TrackMode.Lane0:
            if 5 > ego.x > 3:
                output_vehicle_mode = VehicleMode.SwitchLeft
            if 5 > ego.x > 3:
                output_vehicle_mode = VehicleMode.SwitchRight
    if ego.vehicle_mode == VehicleMode.SwitchLeft:
        if ego.lane_mode == TrackMode.Lane0:
            if ego.x > 10:
                output_vehicle_mode = VehicleMode.Normal
    if ego.vehicle_mode == VehicleMode.SwitchRight:
        if ego.lane_mode == TrackMode.Lane0:
            if ego.x > 10:
                output_vehicle_mode = VehicleMode.Normal

    return output_vehicle_mode, output_lane_mode


from verse.agents.example_agent.car_agent import CarAgent
from verse.scenario.scenario import Scenario
from verse.map.example_map.simple_map import SimpleMap2


class TestSimulatorMethods(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        # Relative path to ../tests/example_controller1.py does not work.
        self.car = CarAgent("ego", file_name="../demo/tacas2023/exp2/example_controller5.py")
        self.car2 = CarAgent("other", file_name="../demo/tacas2023/exp2/example_controller5.py")
        self.scenario.add_agent(self.car)
        self.scenario.add_agent(self.car2)
        tmp_map = SimpleMap2()
        self.scenario.set_map(tmp_map)
        # self.scenario.set_sensor(FakeSensor1())
        # self.scenario.set_init(
        #     [[0, 3, 0, 0.5]],
        #     [(VehicleMode.Normal, TrackMode.Lane0)]
        # )
        # self.traces = scenario.simulate(
        #     10
        # )
        #
        # self.queue = [traces]

    def test_nothing(self):
        self.assertEqual(5, 5)

    def test_carid(self):
        self.assertEqual(self.car.id, "ego", msg="Checking agent creation")

    def test_carinscene(self):
        # Ego agent was added
        self.assertEqual(
            self.scenario.agent_dict["ego"], self.car, msg="Checking adding agent to scenario dict"
        )

    def test_scenedictsize(self):
        # Check size of scene agent dictionary
        self.assertEqual(
            len(self.scenario.agent_dict), 2, msg="Checking adding agent to scenario dict"
        )


if __name__ == "__main__":
    unittest.main()
