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
