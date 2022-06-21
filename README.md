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
The package comes with two controller examples inside folder ```demo/```
- The first example consists a scenario with two vehicles. The second vehicle will brake and stop when it detect the first vehicle in front. The first example can be run by using command

```
python3 demo1.py
```

- The second example consists a scenario with two vehicles, which can perform lane switch based on their relative position.
The second example can be run using command

```
python3 demo2.py
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
