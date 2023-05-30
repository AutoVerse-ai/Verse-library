# Verse Library

Verse is a Python library for creating, simulating, and verifying scenarios with interacting, decision making agents. The decision logic can be written in an expressive subset of Python. The continuous evolution can be described as a black-box simulation function. The agent can be ported across different maps, which can be defined from scratch or imported from [opendrive](https://www.opendrive.com/) files. Verse scenarios can be simulated and verified using hybrid reachability analysis. Verse is developed abd maintained by the [Reliable Autonomy Research Group](https://mitras.ece.illinois.edu/group.html) at the [University of Illinois at Urbana-Champaign](https://ece.illinois.edu/).

<img src="./docs/source/figs/exp1_lab.PNG" height="200"/>

## Installation
The package requires python 3.8+. The package can be installed using pip with all required dependencies

```sh
pip install -e .
```
To update the dependencies in case anything is missing, requirements.txt can be used.

```sh
pip install -r requirements.txt
```

## Tutorial

Interactive Jupyter tutorial: [`tutorial/tutorial.ipynb`](tutorial/tutorial.ipynb).

PDF tutorial: [`tutorial.pdf`](tutorial/tutorial.pdf).

## Demos

See the examples in the `demo/` folder. Run these as:

```sh
python3 demo/ball/ball_bounces.py
```

Read the comments in `demo/ball/ball_bounces.py` to learn how to create new agents and scenarios.

## Using NeuReach with Verse
Verse allows users to plug-in different reachability tools for computing reachable sets. By default, Verse uses DryVR to compute reachable sets. Verse also implement post computation using NeuReach. To use NeuReach, additional dependencies can be downloaded using following commands
```sh
git submodule init
git submodule update
```

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

# Contributors

- Katherine Braught
- Yangge Li
- Sayan Mitra
- Keyi Shen
- Haoqing Zhu
- Daniel Zhuang

# Related Publications

<b> [Verse: A Python library for reasoning about multi-agent hybrid system scenarios](https://arxiv.org/abs/2301.08714)</b>
Yangge Li, Haoqing Zhu, Katherine Braught, Keyi Shen, Sayan Mitra
To appear in the proceedings of Computer Aided Verification (CAV),  2023.

<b> [Verification of L1 Adaptive Control using Verse Library: A Case Study of Quadrotors](https://arxiv.org/abs/2303.13819) </b>
Lin Song, Yangge Li, Sheng Cheng, Pan Zhao, Sayan Mitra, Naira Hovakimyan
To appear in the Work in Progress Session of International Conference on Cyber-Physical Systems (WiP-ICCPS), 2023.

# Contributing

In order to contribute to this repository, you should run the following commands:
```sh
pip install -r requirements-dev.txt
pre-commit install
```
