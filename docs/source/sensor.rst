Sensor
======

Overview
--------

The sensor module defines how agents perceive the world, translating raw state information into
decision logic inputs. This abstraction allows flexible sensor modeling, from omniscient sensors
to realistic limited-range or noisy perception.

Summary
-------

- A Verse sensor reads raw simulation/verification data from each agent.
- It converts that raw data into three dictionaries: ``cont``, ``disc``, and ``len_dict``.
- Your decision logic reads those dictionaries through argument prefixes like ``ego.*`` and ``others.*``.
- In simulation, values are points.
- In verification, values are intervals ``[low, high]`` to represent uncertainty.

Default Behavior: ``BaseSensor``
--------------------------------

By default, all scenarios use the ``BaseSensor`` class, which provides omniscient (perfect) perception
of the environment. This sensor has unrestricted access to:

- **Ego agent**: The agent being controlled, accessed as ``ego`` in the decision logic
- **Other agents**: All other agents in the environment, stored as a list and accessed by argument name
  (``others``) in the decision logic

The ``BaseSensor`` reads the raw state from the state dictionary and automatically extracts continuous
variables, discrete states (modes), and static parameters based on the agent's decision logic definition.
This means no custom perception logic is needed for simple omniscient scenarios (e.g., ``demo\highway\m1_1c1n_dryvr.py``).

Sensor Output Format
~~~~~~~~~~~~~~~~~~~~

All sensors must return three dictionaries via the ``sense()`` method:

1. **Continuous variables dictionary** (``cont``): Maps variable names to their values

   - For ego agent: keyed as ``"ego.var_name"``
   - For other agents: keyed as ``"other.var_name"`` (or other argument names), with values as lists

2. **Discrete variables dictionary** (``disc``): Maps discrete/modal variables to their mode names

   - States must map to the specific mode name (a string), not the mode/class instantiation

3. **Length dictionary** (``len_dict``): Specifies the number of agents in "others" category

   - Must match the actual length of list-valued entries in ``cont`` and ``disc``
   - Example: ``{"others": len(state_dict) - 1}`` when all agents are included

Creating Custom Sensors
-----------------------

While BaseSensor handles omniscient perception, custom sensors can be created to model:

- Limited sensor range
- Measurement noise
- Sensor failures
- Partial/sparse observability
- Multi-sensor fusion

Any custom sensor must implement the following interface:

.. code-block:: python

    class CustomSensor:
        def sense(self, agent, state_dict, lane_map, simulate=True):
            """
            Parameters
            ----------
            agent : BaseAgent
                The agent for which sensing is performed
            state_dict : dict
                Dictionary mapping agent IDs to their state vectors
                Format: {agent_id: [state_vector, modes, static_params], ...}
            lane_map : map
                The scenario map (if applicable)
            simulate : bool
                True for simulation (point values), False for verification (interval bounds).
                Default is True.
                
            Returns
            -------
            cont : dict
                Continuous variables
            disc : dict
                Discrete variables (mode names)
            len_dict : dict
                Number of agents in each category (must match list lengths in cont/disc)
            """
            pass


Understanding Sensor Data Structures
-------------------------------------

Before implementing a custom sensor, understand the input (``state_dict``) and output
(``cont``, ``disc``, ``len_dict``) data structures. **The sensor's core responsibility is to
transform raw** ``state_dict`` **into the three output dictionaries** used by decision logic.

State Dictionary Structure (Input)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``state_dict`` is the raw state representation passed to all sensors. Understanding its structure
is essential for extracting the correct variables when building a custom sensor.

**Structure Overview**

Each agent's state is stored as a list with three elements:

.. code-block:: python

    state_dict = {
        agent_id_1: [state_vector_or_bounds, modes, static_params],
        agent_id_2: [state_vector_or_bounds, modes, static_params],
        ...
    }

**Simulation Mode** (``simulate=True``)
  The sensor provides point values for trajectory simulation:
  
  .. code-block:: python
  
      # Example state_dict in simulation
      state_dict = {
          'robot_1': [[0.5, 1.0, 2.0, 0.3], ['Accelerating'], []],
          'robot_2': [[0.5, 0.5, 0.8, 0.1], ['Cruising'], []]
      }
      # Indices: [0] = state vector, [1] = mode names, [2] = static params (typically empty)
      # State vector format: [time, var_1, var_2, var_3, ...]
  
  To extract variables, skip the time at index 0:
  
  .. code-block:: python
  
      robot1_state = state_dict['robot_1'][0]  # [t, x, y, v]
      x_value = robot1_state[1]  # Skip time, get first variable
      y_value = robot1_state[2]  # Get second variable
      mode = state_dict['robot_1'][1][0]  # Extract first mode name (state_dict['robot'][1] is a list)

**Verification Mode** (``simulate=False``)
    The sensor provides interval bounds for reachsets:
  
  .. code-block:: python
  
      # Example state_dict in verification (reachsets)
      state_dict = {
          'robot_1': [[lower_bounds, upper_bounds], ['Accelerating'], []],
          'robot_2': [[lower_bounds, upper_bounds], ['Cruising'], []]
      }
      # Where lower_bounds and upper_bounds are lists: [time, var_1, var_2, ...] 
      # where lower_bounds[i] <= upper_bounds[i]
  
  To extract interval bounds:
  
  .. code-block:: python
  
      robot1_bounds = state_dict['robot_1'][0]  # [lower, upper]
      lower = robot1_bounds[0]  # [t_lower, x_lower, y_lower, v_lower]
      upper = robot1_bounds[1]  # [t_upper, x_upper, y_upper, v_upper]
      x_interval = [lower[1], upper[1]]  # Skip time (index 0)
      mode = state_dict['robot_1'][1][0]  # 'Accelerating'

Sensor Output (Output Dictionaries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sensor must convert ``state_dict`` into three dictionaries that the decision logic understands:

**Continuous Variables** (``cont``)
  Maps sensor variable names to their values/intervals:
  
  .. code-block:: python
  
      # Simulation mode: point values
      cont = {
          'ego.x': 1.0,
          'ego.y': 2.0,
          'ego.velocity': 0.3,
          'others.x': [0.5, 0.8],  # List: one value per other agent
          'others.y': [0.5, 1.2],
          'others.velocity': [0.1, 0.4]
      }
      
      # Verification mode: intervals
      cont = {
          'ego.x': [0.9, 1.1],  # x given by bounds [0.9, 1.1]
          'ego.y': [1.9, 2.1],
          'ego.velocity': [0.25, 0.35],
          'others.x': [[0.4, 0.6], [0.7, 0.9]],  # List: one intervals per other agent
          'others.y': [[0.4, 0.6], [1.0, 1.4]],
          'others.velocity': [[0.05, 0.15], [0.3, 0.5]]
      }

**Discrete Variables** (``disc``)
  Maps mode names (strings only):
  
  .. code-block:: python
  
      disc = {
          'ego.agent_mode': 'Accelerating',  # String mode name
          'others.agent_mode': ['Cruising', 'Accelerating']  # List of mode names for multiple agents
      }

**Length Dictionary** (``len_dict``)
  Specifies the count of agents in each category. **Must match** the actual length of lists in ``cont`` and ``disc``:
  
  .. code-block:: python
  
    len_dict = {'others': len(state_dict) - 1}  # Must match list lengths in cont/disc

Simulation vs. Verification Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sensors must handle two distinct modes with different data formats:

**Simulation Mode** (``simulate=True``)
  Used during simulation. Returns **point values** for continuous variables.
  
  - Continuous values come from: ``state_dict[agent_id][0][variable_index + 1]``
    (the ``+1`` accounts for time being stored at index 0)
  - Return as: ``cont['var_name'] = value`` (scalar for ego, list for others)

**Verification Mode** (``simulate=False``)
    Used during formal verification with abstract reachsets. Returns **interval bounds**
    for continuous variables.
  
  - Continuous values must be intervals: ``[lower_bound, upper_bound]``
  - Example: ``cont['ego.x'] = [0.5, 1.5]`` means x is between 0.5 and 1.5
  - For multiple agents, use lists of intervals: ``cont['others.x'] = [[0, 1], [2, 3], [1.5, 2.5]]``

Discrete States
~~~~~~~~~~~~~~~

Discrete modes (representing automaton states) must be:

- A **single string** corresponding to the mode name (not the class instantiation)
- Example: ``disc['ego.agent_mode'] = "Accelerating"`` (use a plain string mode name, not a class/enum instance)
- Required for all agents with discrete decision logic
- Must match the name defined in the agent's state definitions

**Critical**: The ``sense`` function must define **all** variables that the decision logic requests
from **all** agents with non-trivial decision logic. If sensor key names do not match what decision
logic expects, verification will fail.


Advanced: Custom Observation Variables
---------------------------------------

For complex scenarios, you can define custom computational states beyond raw states. This allows
the sensor to provide processed, derived, or filtered observations to the decision logic. Instead of 
requiring all agents to report the same raw state variables, you can define custom state classes that 
hold arbitrary computed information—such as distances, relative positions, or filtered measurements.

Defining Extra States in Decision Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom state classes are defined in the decision logic file and represent observed or derived quantities.
They are standard Python classes with typed attributes:

.. code-block:: python

    class ObservedTarget:
        x: float
        y: float
        v: float
        distance: float
        agent_mode: AgentMode # assuming AgentMode is already a defined enum class

        def __init__(self, x, y, v, distance, agent_mode):
            pass

When using lists of these custom observations in a decision logic function:

.. code-block:: python

    def decision_logic(ego: EgoState, others: List[ObservedTarget]):
        # Use reductions over the list of observed targets
        any_threats = any(t.distance < THRESHOLD for t in others)
        all_breaking = all(t.agent_mode == "Brake" for t in others)
        
        if any_threats or (all_breaking and any(t.x < LATERAL_THRESHOLD for t in others)):  # Reductions can appear in conditionals; this is their main use-case
            ...

Sensor Implementation
~~~~~~~~~~~~~~~~~~~~~

In the sensor's ``sense`` method, compute and return the custom observation variables.
For single observations per agent, store directly:

.. code-block:: python

    cont['ego.processed_value'] = value  # Scalar if simulate=True, interval if False

For lists of custom observations (e.g., multiple observed vehicles), store as a list of values in simulation mode or list of intervals in verification mode:

.. code-block:: python

    # For multiple observations of the same variable
    # Each element corresponds to one observed agent
    cont['others.x'] = [0.5, 3.0, -0.25]  # simulation mode
    cont['others.x'] = [[0, 1], [2.5, 3.5], [-1, 0.5]]  # verification mode
    
    # Optionally, store discrete observation states if needed in decision logic
    disc['others.agent_mode'] = ["Accelerating", "Braking", "Cruising"]


Toy Example: Implementing a Custom Sensor
------------------------------------------

Here we give a complete example showing how to transform ``state_dict`` into ``cont`` and ``disc`` dictionaries.
This example demonstrates several practical features: custom observation variables (distance),
noise modeling, and handling both simulation and verification modes.

.. code-block:: python

    import numpy as np
    
    class ToySensor:
        """Sensor that detects other agents and adds noise to some measurements.
        
        This example demonstrates:
        - Computing derived/custom observation variables (e.g., distance)
        - Adding measurement noise
        - Handling both simulation and verification modes
        """
        
        def __init__(self, noise=1e-6):
            self.noise = noise 
        
        def sense(self, agent, state_dict, lane_map, simulate=True):
            # Helper: conservatively compute distance bounds for verification mode
            def get_distance_bounds(ego_interval, other_interval):
                # ego_interval = [x_interval, y_interval] where x_interval = [x_lower, x_upper]
                # Returns [min_distance, max_distance]
                x_ego, y_ego = ego_interval
                x_other, y_other = other_interval
                # Conservative bounds: min dists between closest points, max dist between farthest
                min_dist = np.sqrt((x_ego[0] - x_other[1])**2 + (y_ego[0] - y_other[1])**2)
                max_dist = np.sqrt((x_ego[1] - x_other[0])**2 + (y_ego[1] - y_other[0])**2)
                return [min_dist, max_dist]
            
            cont = {}
            disc = {}
            # len_dict = {"others": len(state_dict)-1}  # Uncomment if populating all other agents
            noise = self.noise

            if simulate:
                # --- SIMULATION MODE: Extract point values ---
                ego_state = state_dict[agent.id][0]  # [t, x, y, theta, v]
                ego_x, ego_y = ego_state[1], ego_state[2]  # Skip time at index 0
                
                # Populate ego state
                cont['ego.x'] = ego_x
                cont['ego.y'] = ego_y
                cont['ego.theta'] = ego_state[3]
                cont['ego.v'] = ego_state[4]
                disc['ego.agent_mode'] = state_dict[agent.id][1][0]
                
                # Populate others state with custom observation variables
                cont['others.x'] = []
                cont['others.y'] = []
                cont['others.v'] = []
                cont['others.distance'] = []  # Custom derived variable
                disc['others.agent_mode'] = []
                
                for other_id in state_dict:
                    if other_id == agent.id:
                        continue
                    
                    other_state = state_dict[other_id][0]
                    other_x, other_y = other_state[1], other_state[2]
                    distance = ((ego_x - other_x)**2 + (ego_y - other_y)**2)**0.5
                    
                    # Add noise to simulated x-position measurements
                    cont['others.x'].append(other_x + np.random.uniform(-noise, noise))
                    cont['others.y'].append(other_y)
                    cont['others.v'].append(other_state[4])
                    cont['others.distance'].append(distance)  # Computed observation
                    disc['others.agent_mode'].append(state_dict[other_id][1][0])
                
                len_dict = {"others": len(cont['others.x'])}
                    
            else:
                # --- VERIFICATION MODE: Extract interval bounds ---
                ego_bounds = state_dict[agent.id][0]
                ego_lower, ego_upper = ego_bounds[0], ego_bounds[1]
                
                # Populate ego intervals [lower, upper]
                cont['ego.x'] = [ego_lower[1], ego_upper[1]]
                cont['ego.y'] = [ego_lower[2], ego_upper[2]]
                cont['ego.theta'] = [ego_lower[3], ego_upper[3]]
                cont['ego.v'] = [ego_lower[4], ego_upper[4]]
                disc['ego.agent_mode'] = state_dict[agent.id][1][0]
                
                # Populate others with conservative bounds and custom observation variables
                cont['others.x'] = []
                cont['others.y'] = []
                cont['others.v'] = []
                cont['others.distance'] = []  # Custom derived variable
                disc['others.agent_mode'] = []
                
                for other_id in state_dict:
                    if other_id == agent.id:
                        continue
                    
                    other_bounds = state_dict[other_id][0]
                    other_lower, other_upper = other_bounds[0], other_bounds[1]
                    
                    # Compute distance interval (accounting for measurement noise)
                    distance_interval = get_distance_bounds(
                        [cont['ego.x'], cont['ego.y']], 
                        [[other_lower[1], other_upper[1]], [other_lower[2], other_upper[2]]]
                    )
                    
                    # Add noise bounds to x-position intervals
                    cont['others.x'].append([other_lower[1] - noise, other_upper[1] + noise])
                    cont['others.y'].append([other_lower[2], other_upper[2]])
                    cont['others.v'].append([other_lower[4], other_upper[4]])
                    cont['others.distance'].append(distance_interval)  # Computed observation
                    disc['others.agent_mode'].append(state_dict[other_id][1][0])
                
                len_dict = {"others": len(cont['others.x'])}
            
            return cont, disc, len_dict

**This example demonstrates:**

1. **Input transformation**: Extracting values from ``state_dict`` and skipping the time dimension
2. **Output dictionaries**: Populating ``cont``, ``disc``, and ``len_dict`` with proper formatting
3. **Custom observation variables**: Computing derived/observed quantities (distance) beyond raw state
4. **Simulation vs Verification**: Different data formats and logic for each analysis mode
5. **Noise modeling**: Adding measurement uncertainty (stochastic in simulation, conservative bounds in verification)


Critical Mapping Rule for Custom Sensors
-----------------------------------------

When implementing custom sensors, dictionary keys must match your decision logic function signature exactly.
This mapping is automatic in ``BaseSensor``, but custom sensors must handle it explicitly.

If your decision logic is:

.. code-block:: python

    def decision_logic(ego: EgoState, others: List[ObservedTarget]):
        ...

Then your sensor must return keys using the **exact parameter names** as prefixes:

- ``cont['ego.x']``, ``cont['ego.y']``, ``disc['ego.agent_mode']``: passed to the ``ego`` parameter
- ``cont['others.x']``, ``cont['others.distance']``, ``disc['others.agent_mode']``: passed to the ``others`` parameter

For list parameters like ``others``, the sensor stores a list of values (e.g., ``cont['others.distance'] = [1.5, 2.3, 0.8]``), 
which the framework packages into a list of objects. In decision logic, you interact with that list
through reductions (``any()`` and ``all()``). Individual list elements are not exposed directly,
which keeps the interface compatible with the parser.


Integration
-----------

To use a custom sensor in a scenario file (assuming the sensor is defined in the same directory):

.. code-block:: python

    from custom_sensor import CustomSensor
    
    # Create scenario
    scenario = Scenario(ScenarioConfig(...))
    
    # Instantiate and add the sensor
    # Note: CustomSensor can accept initialization parameters if needed
    custom_sensor = CustomSensor(noise=1e-4)
    scenario.set_sensor(custom_sensor)
    
    # Now run verification or simulation
    scenario.verify() # or scenario.simulate()


Example Reference Implementations
----------------------------------

For complete examples of custom sensor implementation, see:

- ``verse/sensor/base_sensor.py``: Default omniscient sensor
- ``tutorial/tutorial_sensor.py``: Tutorial sensor example
- ``demo/aprod/vis_dubins_rdvz/car_sensor_parser_3.py``: Advanced sensor with custom processing

See ``demo/aprod/vis_dubins_rdvz/controller_3.py`` for
a complete example of custom observation handling.
