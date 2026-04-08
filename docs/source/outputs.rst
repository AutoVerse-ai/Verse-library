Analysis
~~~~~~~~

Verse provides several functions for analyzing hybrid systems defined by scenarios. 

The ``simulate()`` function generates simulation traces of scenarios. The results of simulation can be visualized 
using :ref:`Visualization` functions or stored as a json file using the ``dump()`` function in the following format::

	"ID": {
		"agent": {<list of agents>},
		"assert_hits": <result>,
		"child": [<list of node IDs>],
		"id": int,
		"init": {<list of initial state>},
		"mode": {<list of modes>},
		"parent": <parent node id>,
		"start_time": float,
		"static": ,
		"trace": {<list of traces for agents>},
		"type": "simtrace"
	},


Each trace in the list of traces is of the form::

	{
		"<agent_name>": [
			[time,
			x1,
			x2,
			x3
			], ...
		]
	}

Reachtubes
~~~~~~~~~~

Verse likewise provides the `verify()` function to compute reachtubes (i.e., over-approximations of the set of reachable sets of a time interval).
Verification (reachtube) results are stored in the same dump-style JSON tree used for simulation outputs, but nodes contain reachsets (reachtubes) for each agent over time. 
A typical node in a verification output looks like::

	{
	    "0": {
	        "agent": {
	            "car1": "<class 'verse.agents.example_agent.car_agent.CarAgent'>",
	            "car2": "<class 'verse.agents.example_agent.car_agent.NPCAgent'>"
	        },
	        "assert_hits": null,
	        "child": [1, 2],
	        "height": 0,
	        "id": 0,
	        "init": { ... },
	        "mode": { ... },
	        "ndigits": 10,
	        "parent": null,
	        "start_time": 0,
	        "static": { ... },
	        "trace": {
	            "car1": [
	                [0.0, 0.0, -0.5, -0.006418150078359816, 1.0],
	                [0.05, 0.061011626236203695, 0.5, 0.006418150078359816, 1.0],
	                [0.05, 0.04898837376379629, -0.48914532308843445, -0.012507427337564252, 1.0],
	                [0.1, 0.11197990465794036, 0.48914532308843445, 0.012507427337564252, 1.0],
	                ...
	            ]
	        }
	    }
	}

Notes:

- This is the exact same format as in the simulation dump.
- ``agent``: mapping of agent names to their class identifiers.
- ``init`` / ``mode`` / ``static``: initial sets, modes, and static parameters for each agent.
- ``trace``: the reachtube data for each agent. Each entry is a list of vectors of the form ``[time, x1, x2, ...]`` representing the reachset/interval at that sample.
- This section uses the highway demo found at `demo/highway/m1_1c1n/m1_1c1n_dryvr_ver.py` as a running example. The demo sets up two agents (`car1`, `car2`) with interval initial states and modes, assigns the map `M1()`, uses a time step of 0.05 s, and invokes verification via `scenario.verify(40, 0.05)`. See [demo/highway/m1_1c1n/m1_1c1n_dryvr_ver.py](demo/highway/m1_1c1n/m1_1c1n_dryvr_ver.py#L1) for the full script.

Accessing a reachtube in code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a verification call such as::

	traces = scenario.verify(T, ts)

you can access the reachtube (trace) for the first node and an agent named ``car1`` like this::

	traces.nodes[0].trace['car1']

This returns the list of time-tagged reachset vectors shown above.

Hyperrectangle ordering and repeated times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: reachtube outputs that return hyperrectangles use an alternating ordering of lower/upper corner vectors. By convention the even-indexed entries (indices where `i % 2 == 0`) are the lower bounds and the immediately following odd entries (`i + 1`) are the corresponding upper bounds. Consequently, for every even index `i` and every dimension `j`:

``trace[i][j] <= trace[i+1][j]``

Because entries are stored as paired hyperrectangles per sample, the time field may repeat between adjacent entries if one is using DryVR (for example, you may see two consecutive entries with the same timestamp, representing the lower and upper corner at that sample). Example sequence::

	[0.0, ...], [0.05, ...], [0.05, ...], [0.1, ...]

In the example above the two `0.05` entries are the lower and upper corners for the same sample time.

Quick code pattern
~~~~~~~~~~~~~~~~~~

To iterate the per-sample hyperrectangles and access lower/upper corners use::

	trace = traces.nodes[0].trace['car1']
	for k in range(0, len(trace), 2):
		lower = trace[k]
		upper = trace[k+1]
		# lower[d] <= upper[d] for all dimensions d
		# timestamps: lower[0] == upper[0] (may repeat)

Use this section as the reference for reading verification dumps and for programmatic access to reachtube data.
