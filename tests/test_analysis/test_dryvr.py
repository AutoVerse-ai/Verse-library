
import unittest
import numpy as np

from verse.analysis.dryvr import all_sensitivities_calc


training_traces_example = np.array([
    [[1, 2, 3, 4], 
     [1, 2, 3, 4]],
    [[1, 2, 3, 4], 
     [3, 4, 5, 6]]
])

initial_radii = np.array([0, 0, 1])

class testDryVR(unittest.TestCase):
    def test_all_sensitivities_calc(self):
        print(training_traces_example.shape)
        print(all_sensitivities_calc(training_traces_example, initial_radii))

if __name__ == "__main__":
    # unittest.main()
    # print("DryVR reachability engine test done")
    res = all_sensitivities_calc(training_traces_example, initial_radii)
    print(res)