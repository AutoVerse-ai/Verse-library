.. verse documentation master file, created by
   sphinx-quickstart on Fri Jul 29 22:19:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Verse documentation!
===============================

Verse is a Python library for creating, simulating, and verifying scenarios with interacting,  agents developed at the `Reliable Autonomy Research Group <https://mitras.ece.illinois.edu/group.html>`_ of University of Illinois, Urbana-Champaign. The decision logic of an agent can be written in an expressive subset of Python. The continuous evolution can be described as a black-box simulation function. The agent can be ported across *maps*, which can be defined from scratch or imported from opendrive. Verse scenarios can be simulated and verified using hybrid reachability analysis. For a technical overview see the  `Verse paper from CAV23 <https://link.springer.com/chapter/10.1007/978-3-031-37706-8_18>`_ or `arXiv <https://arxiv.org/abs/2301.08714>`_.

.. image:: figs/car-ped-1.png
   :height: 150
   :alt: Car pedestrian interaction
.. image:: figs/drone-2-8.gif
   :height: 150
   :alt: Drones on tracks
.. image:: figs/nondeterm_sensor.png
   :height: 150
   :alt: Nondeterministic sensors
.. image:: figs/uam-collision.png
   :height: 150
   :alt: Collision avoidance in urban air mobility


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started.rst
   creating_scenario_in_verse.md
   troubleshooting.md
   agent.md
   map.rst
   sensor.rst
   scenario.rst
   parser.md
   outputs.rst
   plotting.rst
   contributors.rst
   publications.rst


API Documentation
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   verse

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
