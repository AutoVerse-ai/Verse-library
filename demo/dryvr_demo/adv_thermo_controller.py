from enum import Enum, auto
import copy


class ThermoMode(Enum):
    WARM = auto()
    WARM_FAST = auto()
    COOL = auto()
    COOL_FAST = auto()


class State:
    temp = 0.0
    total_time = 0.0
    cycle_time = 0.0
    thermo_mode: ThermoMode = ThermoMode.WARM

    def __init__(self, temp, total_time, cycle_time, thermo_mode: ThermoMode):
        pass


def decisionLogic(ego: State):
    output = copy.deepcopy(ego)
    if ego.thermo_mode == ThermoMode.WARM or ego.thermo_mode == ThermoMode.WARM_FAST:
        if 1.1 >= ego.cycle_time >= 1.0:
            # if ego.temp >= 83:
            output.thermo_mode = ThermoMode.COOL_FAST
            if True:
                output.thermo_mode = ThermoMode.COOL
            output.cycle_time = 0.0
    if ego.thermo_mode == ThermoMode.COOL or ego.thermo_mode == ThermoMode.COOL_FAST:
        if 1.1 >= ego.cycle_time >= 1.0:
            # if ego.temp <= 72:
            output.thermo_mode = ThermoMode.WARM
            if True:
                output.thermo_mode = ThermoMode.WARM_FAST
            output.cycle_time = 0.0
    return output
