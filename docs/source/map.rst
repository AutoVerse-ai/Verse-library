Maps
====

Overview
--------

The map of a scenario specifies the tracks that agents can follow and provides helper functions to inspect and
modify track-based information (for example, which track an agent should follow after a mode switch).

This page gives a brief introduction and points to the tutorial and demo scrips for runnable examples.

Basic usage
-----------

Import a predefined map and attach it to a scenario::

	from tutorial_map import M3
	from verse.scenario import Scenario

	scenario = Scenario()
	map1 = M3()
	scenario.set_map(map1)

In the tutorial notebook the simple two-track map ``M3`` is used to demonstrate track modes ``T0`` and ``TAvoidUp``.
See the tutorial for a runnable walkthrough: `tutorial notebook <https://github.com/AutoVerse-ai/Verse-library/blob/main/tutorial/tutorial.ipynb>`_.

Track-mode helpers
------------------

Maps expose helper functions used by agents' decision logic to pick new track modes when a tactical mode
changes. Two commonly used helpers are:

- ``h(current_track_mode, current_agent_mode, next_agent_mode)``: returns a new track mode for the agent
  after a tactical-mode transition.
- ``h_exist(current_track_mode, current_agent_mode, next_agent_mode)``: boolean check whether a
  transition is defined.

Example (referenced in the tutorial): the tutorial shows how to call ``h_exist`` and ``h`` from decision
logic to test and select valid mode transitions (see Section 1.1 and Section 5 of the
`tutorial notebook <https://github.com/AutoVerse-ai/Verse-library/blob/main/tutorial/tutorial.ipynb>`_ for examples).

Advanced maps
-------------

More advanced maps can define multiple lanes, concatenated lane segments, altitude/geometry helpers, and
custom transition rules. See the tutorial (Section 5) for an extended example that builds a 2-lane map,
defines transition modes such as ``M01``/``M10``, and demonstrates how to use the map's ``h`` and
``h_exist`` functions in practice.

OpenDRIVE maps
^^^^^^^^^^^^^^

To created said advanced road geometries, Verse supports importing
ASAM OpenDRIVE (.xodr) files. The repository includes a parser at
`verse/map/opendrive_parser.py <https://github.com/AutoVerse-ai/Verse-library/blob/main/verse/map/opendrive_parser.py>`_ that converts XODR road elements into
`Lane` / `LaneMap` objects, computes lane widths, and populates transition rules (the ``h`` /
``h_exist`` helpers) automatically.

For reference, we refer users to the demo that uses OpenDRIVE input for an example workflow: `demo script <https://github.com/AutoVerse-ai/Verse-library/blob/main/demo/highway/m4_1c2n/m4_1c2n.py>`_. In that demo the parser is invoked on an example file
(`https://github.com/AutoVerse-ai/Verse-library/blob/main/demo/highway/m4_1c2n/t1_triple.xodr`) and the resulting LaneMap is attached to the scenario via
``scenario.set_map(...)``.

If one needs to create or import realistic road networks (curves, merges, many lanes), we recommend checking the built-in OpenDRIVE parser
`verse/map/opendrive_parser.py <https://github.com/AutoVerse-ai/Verse-library/blob/main/verse/map/opendrive_parser.py>`_ as a starting point.

The ASAM OpenDRIVE standard and reference information is available from the official site: `ASAM OpenDRIVE <https://www.asam.net/standards/detail/opendrive/>`_.
