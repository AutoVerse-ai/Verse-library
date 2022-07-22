# DryVR++
## Installation
The package requires python 3.8+. The package can be installed using pip

```
python3 -m pip install -e .
```
To update the dependencies, setup.py or requirement.txt can be used.

```
python3 setup.py install
```
or
```
pip install -r requirements.txt
```

## Examples
The package comes with several examples in the  ```demo/``` folder
- Run examples as:

```
python3 demo1.py
```

Read the comments in ```ball_bounces.py``` to learn how to create new agents and scenarios. More detailed tutorials will be provided later.

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
