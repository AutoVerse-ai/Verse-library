import unittest
from verse.sensor import BaseSensor

# TODO: Need some discussion about how to effectively test the sense function in the base_sensor.py code in sensor directory
# As there are some helper functions (add_2d_states) return errors when accessing the agent as in the sense function.
# it would be better to have further discussion on how to test this function effectively