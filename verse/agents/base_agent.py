from verse.parser.parser import ControllerIR

class BaseAgent:
    """
        Agent Base class

        Methods
        -------
        TC_simulate
    """
    def __init__(self, id, code = None, file_name = None): 
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
        self.controller: ControllerIR = ControllerIR.parse(code, file_name)
        self.id = id

    def TC_simulate(self, mode, initialSet, time_horizon, time_step, map=None):
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
        raise NotImplementedError
