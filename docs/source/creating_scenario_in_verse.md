# Creating a Scenario
This is an introduction to creating scenarios in Verse. An interactive tutorial with more details are in [this Jupyter notebook](https://github.com/AutoVerse-ai/Verse-library/blob/main/tutorial/tutorial.ipynb).

A Verse *scenario* is defined by a *map*, a set of *agents*, and optionally  a *sensor*. We will create a  scenario with a drone following a straight path that dodges obstacles by moving up or down.

## Import a map
A *map* specifies the *tracks* or paths that the agents *is allowed to follow*. Our map will have two kinds of tracks: 

    1. <code>T0</code> is an x-axis aligned track 
    2. <code>TAvoidUp</code> is a upward track for avoiding obstacles on the x-axis. 

<code>T0</code> and <code>TAvoidUp</code> are called the *track modes* in  Verse. To create new maps of your own, see {doc}`Map<map>`. For now, import a pre-defined map with:

```python
from tutorial_map import M3

map1 = M3()
```

> ####  **Notice**
> Set <code>PYTHONPATH</code> if necessary to include the Verse directory. For example:
> ```shell
> export PYTHONPATH=../../Verse-library
>```

## Create an agent
An *agent* is defined by:

1. Set of *tactical modes* which define the kinds of behavior the agent *wants to* perform. Tactical and track modes together define the discrete *modes* of the agent. 
2. Set of *state variables* that define the continuous state of the agent in the map or the physical world, as well as the discrete modes.
3. *Decision logic* which define mode changes. 
4. *Flow function* which defines continuous variable changes. 

In our scenario, the tactical modes will be <code>Cruise</code> and <code>Up</code>. The decision logic also need to know the available track modes from the map. The tactical modes and track modes are provided as <code>Enums</code> to Verse.

```python
from enum import Enum, auto


class CraftMode(Enum):
    Cruise = auto()
    Up = auto()


class TrackMode(Enum):
    T0 = auto()
    TAvoidUp = auto()
```

Define the state variables in the <code>State</code> class. You can name your variables however you like. Variables names ending with <code>_mode</code> will be identified as  discrete variables. Here <code>craft_mode</code> and <code>track_mode</code> are the discrete variables and the types are necessary to associate them with the  tactical and track modes.

```python
class State:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    craft_mode: CraftMode
    track_mode: TrackMode
```

The *decision logic* is a function that takes as input the agent's current state (and optionally the observable states of the other agents) and returns the new state after a transition. That is, it  updates the tactical mode of the agent. The drone starts in <code>Cruise</code> mode and the obstacle is at 20 meters. When the x position of the drone is close 20, the decision logic switches to <code>Up</code>. 

```python
import copy

def decisionLogic(ego: State, track_map):
    next = copy.deepcopy(ego)
    if ego.craft_mode == CraftMode.Normal:
        if ego.x > 20:
            next.craft_mode = CraftMode.Up
            next.track_mode = track_map.map2track(ego.track_mode, ego.craft_mode, CraftMode.Up)
    return next
```

What is <code>track_map.map2track</code>, you might wonder. Recall, tracks are defined by the map, and when the agent decides to change its behavior, say to move up, then the actual path it can follow is given by the map and this is computed using the <code>track_map.map2track</code> function. 

We incorporate the above definition of tactical modes and decision logic into code strings and combine it with an imported agent flow, we can then obtain the agent for this scenario.

```python
from tutorial_agent import DroneAgent

drone1 = DroneAgent("drone1", file_name="dl_sec1.py", t_v_pair=(1, 1), box_side=[0.4] * 3)
```

More details about how agents works in Verse are described in {doc}`Agents<agent>`

## Creating Scenario
With the agent and map defined, we can now define the scenario.

```python
from verse.scenario import Scenario

scenario = Scenario()
```

We can set the initial condition of the agent and add the agent

```python
drone1.set_initial(
    [[0, -0.5, -0.5, 0, 0, 0], [1, 0.5, 0.5, 0, 0, 0]], (CraftMode.Normal, TrackMode.T0)
)
scenario.add_agent(drone1)
```

and set the map for the scenario
```python
scenario.set_map(map1)
```

Since we only have one agent in the scenario, we don't need to specify a sensor.

We can then compute simulation traces or reachable states for the scenario
```python
traces_simu = scenario.simulate(60, 0.2)
traces_veri = scenario.verify(60, 0.2)
```

We can visualize the results using functions provided with Verse

```python
from verse.plotter.plotter3D import *
import pyvista as pv
import warnings

warnings.filterwarnings("ignore")
from verse.plotter.plotter2D import *
import plotly.graph_objects as go

fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 1, 3)
fig.show()

pv.set_jupyter_backend(None)
fig = pv.Plotter()
fig = plot3dMap(map1, ax=fig)
fig = plot3dReachtube(traces_veri, "drone1", 1, 2, 3, color="r", ax=fig)
fig.set_background("#e0e0e0")
fig.show()
```

The visualized result looks like

z vs t plot             |  x,y,z plot
:-------------------------:|:-------------------------:
![](figs/newplot.png)     |  ![](figs/output.png)
