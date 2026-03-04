from bounded_piecewise import bounded_piecewise_z3, parse_piecewise_function
from z3 import *

if __name__ == "__main__":
    def vis_sensor_piecewise(psi, phi):
        if psi/(phi+1) >= 0.5 and psi <= phi:
            return -1
        elif psi < 0 and psi >= -phi:
            return 1
        elif psi > phi:
            # return -2 
            return something(psi, 2) # parser should be able to parse out arbitrary functions
        else:
            return 2

    # Parse it
    conditions_outputs = parse_piecewise_function(vis_sensor_piecewise)

    # print(conditions_outputs)
    # Use it
    input_bounds = {
        'psi': [-1.1, 2],
        'phi': [1, 1]
    }

    z3_vars = {'psi': Real('psi'), 'phi': Real('phi')}
    results = bounded_piecewise_z3(conditions_outputs, input_bounds)

    for constrained_bounds, output_func in results:
        print(f"Constrained bounds: {constrained_bounds}")
        print(f"Output function: {output_func}")