from verse.parser.parser import ControllerIR
import numpy as np
from scipy.integrate import ode
import copy


class BaseAgent:
    """
    Agent Base class

    Methods
    -------
    TC_simulate
    """

    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state = None, 
        initial_mode = None
    ):
        """
        Constructor of agent base class.

        Parameters
        ----------
        id : str
            id of the agent.
        code: str
            actual code of python controller
        file_name: str
            file name to the python controller
        """
        self.decision_logic: ControllerIR = ControllerIR.parse(code, file_name)
        self.id = id
        self.init_cont = initial_state
        self.init_disc = initial_mode
        self.static_parameters = None 
        self.uncertain_parameters = None

    def set_initial(self, initial_state, initial_mode, static_param=None, uncertain_param=None):
        '''Initialize the states
        '''
        self.set_initial_state(initial_state)
        self.set_initial_mode(initial_mode)
        self.set_static_parameter(static_param)
        self.set_uncertain_parameter(uncertain_param)

    def set_initial_state(self, initial_state):
        self.init_cont = copy.deepcopy(initial_state)

    def set_initial_mode(self, initial_mode):
        self.init_disc = copy.deepcopy(initial_mode)

    def set_static_parameter(self, static_param):
        self.static_parameters = copy.deepcopy(static_param)

    def set_uncertain_parameter(self, uncertain_param):
        self.uncertain_parameters = copy.deepcopy(uncertain_param)

    @staticmethod
    def dynamic(t, state, u):
        raise NotImplementedError()

    def TC_simulate(self, mode, initialSet, time_horizon, time_step, map=None):
        # TODO: P1. Should TC_simulate really be part of the agent definition or
        # should it be something more generic?
        # TODO: P2. Looks like this should be a global parameter;
        # some config file should be setting this.
        """
        Abstract simulation function

        Parameters
        ----------
            mode: str
                The current mode to simulate
            initialSet: List[float]
                The initial condition to perform the simulation
            time_horizon: float
                The time horizon for simulation
            time_step: float
                time_step for performing simulation
            map: LaneMap, optional
                Provided if the map is used
        """
        num_points = int(np.ceil(time_horizon / time_step))
        trace = np.zeros((num_points + 1, 1 + len(initialSet)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = initialSet
        for i in range(num_points):
            result = ode(self.dynamic)
            result.set_initial_value(initialSet)
            res: np.ndarray = result.integrate(result.t + time_step)
            initialSet = res.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = initialSet
        return np.array(trace)
