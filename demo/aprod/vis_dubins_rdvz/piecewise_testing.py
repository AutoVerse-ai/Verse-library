from bounded_piecewise import bounded_piecewise_z3
from z3 import *

def example_vis_sensor(input_bounds):
    """
    if psi >= 0 and psi <= phi: return -1
    elif psi < 0 and psi >= -phi: return 1
    elif psi > phi: return -2
    else: return 2
    """
    
    z3_vars = {
        'psi': Real('psi'),
        'phi': Real('phi')
    }
    
    ### NOTE: two hacks here 
    # 1. converted all > and < to >= and <= (treating infinium as minimum and same for suprenum)
    # 2. explicitly labeled else clause
    conditions_outputs = [
        (And(z3_vars['psi'] >= 0, z3_vars['psi'] <= z3_vars['phi']), 
         lambda v: -1),
        
        (And(z3_vars['psi'] < 0, z3_vars['psi'] >= -z3_vars['phi']), 
        # (And(z3_vars['psi'] <= 0, z3_vars['psi'] >= -z3_vars['phi']), 
         lambda v: 1),
        
        (z3_vars['psi'] > z3_vars['phi'], 
        # (z3_vars['psi'] >= z3_vars['phi'], 
         lambda v: -2),
        
        (None,  # else clause
         lambda v: 2),
        # (z3_vars['psi'] <= -z3_vars['phi'],
        #  lambda v: 2),
    ]
    
    results = bounded_piecewise_z3(conditions_outputs, input_bounds)
    
    for constrained_bounds, output_func in results:
        print(f"Constrained bounds: {constrained_bounds}")
        print(f"Output: {output_func(constrained_bounds)}")

if __name__ == "__main__":
    input_bounds = {
        'psi' : [-1.1,2], 
        'phi' : [1,1]
        }
    int_results = example_vis_sensor(input_bounds)