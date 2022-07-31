# Verse Core Library

Verse is a library for creating, simulating, and verifying uni*verses* or scenarios with intelligent and interacting autonomous agents.  

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

## Demos
The package comes with several examples in the  ```demo/``` folder. Run these as:

```
python3 demo/vehicle/demo2.py 
```

Read the comments in ```demo/ball/ball_bounces.py``` to learn how to create new agents and scenarios. More detailed tutorials will be provided later.

## Library structure

The source code of the package is contained in the verse folder, which contains the following sub-directories.

- **verse**, which contains building blocks for creating and analyzing scenarios.
  
  - **verse/scenario** contains code for the scenario base class. A scenario is constructed by several **agents** with continuous dynamics and controller, a **map** and a **sensor** defining how different agents interact with each other.
  - **verse/agents** contains code for the agent base class in the scenario. 
  - **verse/map** contains code for the lane map base class and corresponding utilities in the scenario.
  - **verse/code_parser** contains code for converting the controller code to ASTs. 
  - **verse/automaton** contains code implementing components in hybrid-automaton
  - **verse/analysis** contains the **Simulator** and **Verifier** and related utilities for doing analysis of the scenario
  - **verse/dryvr** dryvr for computing reachable sets


- **example** contains example map, sensor and agents that we provided


- **plotter** contains code for visualizing the computed results
