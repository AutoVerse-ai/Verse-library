from z3 import *
from typing import List, Tuple, Dict

from z3 import *
from typing import List, Tuple, Dict

def z3num_to_float(x):
    # Simplify expression (handles things like "-1 * epsilon")
    x = simplify(x)

    # Numeric types: Int, Rational, Algebraic
    if isinstance(x, (IntNumRef, RatNumRef, AlgebraicNumRef)):
        s = x.as_string().rstrip('?')
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)

    # Symbolic expressions involving epsilon → replace epsilon → 0
    s = str(x).replace("epsilon", "0")

    # The cleaned string may contain spaces/operators; extract the number:
    try:
        return float(eval(s))   # safe because s only contains 0, digits, +, -, *, /
    except:
        raise ValueError(f"Cannot parse Z3 numeric expression: {x}")
    
def bounded_piecewise_z3(conditions_outputs, input_bounds):
    """
    Evaluate a piecewise function over bounded inputs using Z3.
    """
    
    results = []
    var_names = list(input_bounds.keys())
    
    # Create Z3 variables
    z3_vars = {name: Real(name) for name in var_names}
    
    # Track which conditions have been checked (for else clause)
    negated_conditions = []
    
    for i, (condition, output_func) in enumerate(conditions_outputs):
        # Determine the condition for this branch
        if condition is None:
            # Else clause: negate all previous conditions
            if negated_conditions:
                branch_condition = And(negated_conditions)
            else:
                branch_condition = BoolVal(True)
        else:
            branch_condition = condition
            negated_conditions.append(Not(condition))
        
        # Branch is satisfiable, now find constrained bounds
        constrained_bounds = {}
        
        for name in var_names:
            # Find minimum value that satisfies both input bounds AND branch condition
            opt_min = Optimize()
            for bound_name, (min_val, max_val) in input_bounds.items():
                opt_min.add(z3_vars[bound_name] >= min_val)
                opt_min.add(z3_vars[bound_name] <= max_val)
            opt_min.add(branch_condition)
            # opt_min.minimize(z3_vars[name])
            obj = opt_min.minimize(z3_vars[name])

            min_val = None
            if opt_min.check() == sat:
            #     model = opt_min.model()
            #     min_val_expr = model[z3_vars[name]]
            #     min_val = float(min_val_expr.as_decimal(10))
                # Extract INFIMUM of the objective
                inf = opt_min.lower(obj)       # <-- This is the correct API for Optimize
                min_val = z3num_to_float(inf)

            # Find maximum value that satisfies both input bounds AND branch condition
            opt_max = Optimize()
            for bound_name, (min_val_b, max_val_b) in input_bounds.items():
                opt_max.add(z3_vars[bound_name] >= min_val_b)
                opt_max.add(z3_vars[bound_name] <= max_val_b)
            opt_max.add(branch_condition)
            # opt_max.maximize(z3_vars[name])
            obj = opt_max.maximize(z3_vars[name])
            

            max_val = None
            if opt_max.check() == sat:
                sup = opt_max.upper(obj)       # <-- This is the correct API for Optimize
                max_val = z3num_to_float(sup)
                # model = opt_max.model()
                # max_val_expr = model[z3_vars[name]]
                # max_val = float(max_val_expr.as_decimal(10))
            
            if min_val is not None and max_val is not None:
                constrained_bounds[name] = (min_val, max_val)
        
        # Add result if we got bounds for all variables
        if len(constrained_bounds) == len(var_names):
            results.append((constrained_bounds, output_func))
    
    return results

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