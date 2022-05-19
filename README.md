# DryVR++
## Installation
The package requires python 3.8+. All the required packages can be installed through

```
python3 -m pip install -r requirements.txt
```

## Examples
The package comes with two controller examples
- The first example consists a scenario with a single vehicle, which can perform lane switch based on its location. The 
first example can be run by using command

```
python3 example_car_lane_switch.py
```

- The second example consists a scenario with two vehicles, which can perform lane switch based on their relative position.
The second example can be run using command

```
python3 example_two_car_lane_switch.py
```

## Package Structure

The source code of the package is contained in the src folder, which contains the following sub-directories.

- **scene_verifier**, which contains building blocks for creating and analyzing scenarios.
  
  - **scene_verifier/scenario** contains code for the scenario base class. A scenario is constructed by several **agents** with continuous dynamics and controller, a **map** and a **sensor** defining how different agents interact with each other.
  - **scene_verifier/agents** contains code for the agent base class in the scenario. 
  - **scene_verifier/map** contains code for the lane map base class and corresponding utilities in the scenario.
  - **scene_verifier/code_parser** contains code for converting the controller code to ASTs. 
  - **scene_verifier/automaton** contains code implementing components in hybrid-automaton
  - **scene_verifier/analysis** contains the **Simulator** and **Verifier** and related utilities for doing analysis of the scenario
  - **scene_verifier/dryvr** dryvr for computing reachable sets


- **example** contains example map, sensor and agents that we provided


- **plotter** contains code for visualizing the computed results
