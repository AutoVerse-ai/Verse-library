# Agent 

As mentioned in {doc}`creating scenario in Verse<creating_scenario_in_verse>`, a scenario in Verse is composed by multiple agents. Agents in Verse can be instantiated by a `simulator` and a `decisionLogic`. An example for instantiating a car agent for Verse is shown below.  
```
agent = CarAgent(
    "car1",
    file_name=input_code_name,
    initial_state=[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]],
    initial_mode=(AgentMode.Normal, TrackMode.T1),
)
``` 
The agent can then be added to the scenario using the `scenario.add_agent` function as 
```
scenario.add_agent(agent)
```

## Agent Class
The agent in Verse is represented by the agent class. In the example above it's the `CarAgent` class. To instantiate an agent in Verse, the user need to provide 
1) A unique identifier for the agent in `str`, in this example it's `"car1"`. Different agents in the same scenario should have different identifier.
2) The decision logic of the agent. In this example, the decision logic is provided by `input_code_name`, which contains the path to the python file that contains the decision logic. Details about the decision logic will be described in following sections
3) The set of initial continuous states for the agent. Currently, Verse only accepts initial states given in boxes. In this example the inital_state is given as `[[0, -0.5, 0, 1.0], [0.01, 0.5, 0, 1.0]]` with the first list providing the lower bound for the four continuous states of the car and the second list providing the upper bound.
4) The initial mode of the agent. The set of possible mode is described in the decision logic. In this example, the initial mode is given by `(AgentMode.Normal, TrackMode.T1)`.

Currently, all example agent classes in Verse is derived from the [BaseAgent](https://github.com/AutoVerse-ai/Verse-library/blob/main/verse/agents/base_agent.py) class. Detailed explaination for `BaseAgent` and how to create custom agent will be discussed in {doc}`creating custom agent<create_custom_agent>`

## Simulator
The simulator is an essential part of the agent which describe how the continuous state evolve for the agent. The simulator is used in both hybrid simulation and verification in Verse. The simulator for the agent have to be implemented in the `TC_simulate` function in the agent class. The `TC_simulate` function takes the following inputs
- `mode`:`str`, which is the current mode of the agent 
- `initialState`:`List[float]`, which is the initial continuous states to perform the simulation
- `time_horizon`: `float`, the time horizon for simulation
- `time_step`: `float`, the time step for where the simulation is evaluated
- `map`: `LaneMap`, optional, provided if the map is used 

and it will return an `numpy.ndarray`, which contains the simulation trajectory of the system starting from `initialState` untinal `time_horizon` and evaluated every `time_step`. The dimension of the result will be Tx(N+1) where T is the number time points that the simulator evaluates and N+1 is time and the number of dimensions of the agent. 

## DecisionLogic
The decision logic describes how the discrete mode of the agent can change. 
It's a piece of executable Python code. The decision logic can be provided to the agent by either specifying the path to the `.py` file that contains the decision logic or by providing the code string containing the decision logic directly. The decision logic used in this example can be found [here](https://github.com/AutoVerse-ai/Verse-library/blob/main/demo/highway/m1_1c1n/example_controller4.py). The main function in the decision logic is the `decisionLogic` function (have to with this name).
It takes as input the current state of the agent, the state of all other agents and the map, and output the updated mode of the current agent and potentially perform some modification to the continuous mode of the current agent. 
More details about how to create a decision logic can be found in {doc}`creating decision logic<create_decision_logic>`
