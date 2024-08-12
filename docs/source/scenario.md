# Scenario

In Verse, a scenario represents a certain environment in which verification experiments can happen, and thus would be the main object for interaction. As mentioned in [Creating Scenario in Verse](creating_scenario_in_verse.md), a scenario mainly consists of 3 components:
1. Agents which can interact with one another,
2. A map in which these agents interact, and defines apart of the dynamics of these agents, and
3. A sensor that controls how the state information of agents will be shared.

When no sensor is explicitly supplied, the default sensor will be used, which simply exposes all state information of all agents as-is to all agents.

In addition to these components, a scenario can be configured using a `ScenarioConfig` object, which controls the simulation and verification behaviors.

```{eval-rst}
.. currentmodule:: verse
.. autosummary::
    :toctree: _autosummary

    ScenarioConfig
    Scenario.add_agent
    Scenario.set_init_single
    Scenario.set_init
    Scenario.set_sensor
    Scenario.set_map
```

## Simulation & Reachability Analysis

Once a scenario is created, the user can perform simulation and verification in the scenario. This is done by calling either the `.simulate()` or `.verify()` methods. The common parameters for both methods are:

- `time_horizon`: The length of time to run the simulation/reachability analysis for, in seconds.
- `time_step`: The length of time between discrete transitions are checked, in seconds. This is also the time resolution at which trajectories are recorded. As the `time_step` gets smaller, the analysis will get more accurate, but also take more time to finish.
- `max_height`: The height at which the analysis will stop for a particular branch/path.

```{eval-rst}
.. currentmodule:: verse
.. autosummary::
    :toctree: _autosummary

    Scenario.simulate
    Scenario.verify
```

Both methods produce traces of type `AnalysisTree`. An `AnalysisTree` stores the segments of trajectories a scenario produced, represented by an `AnalysisTreeNode`. A new node or trajectory segment is generated as the mode of the system changes. Note that as a system may have non-deterministic transitions, hence why the resulting reachset is a tree.

```{eval-rst}
.. currentmodule:: verse.analysis
.. autosummary::
    :toctree: _autosummary

    AnalysisTree
    AnalysisTreeNode
```
