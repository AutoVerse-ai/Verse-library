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

## DecisionLogic