import itertools, copy
import numpy as np

from verse.parser import unparse

class ResetExpression():
    def __init__(self, reset):
        reset_var, reset_val_ast = reset 
        self.var = reset_var
        self.val_ast = reset_val_ast
        self.expr = unparse(reset_val_ast)

    def __eq__(self, o) -> bool:
        if o is None:
            return False 
        if not isinstance(o, ResetExpression):
            return False 
        return self.var == o.var and self.expr == o.expr

# class ResetExpression:
#     def __init__(self, reset_list):
#         self.ast_list = []
#         for reset in reset_list:
#             self.ast_list.append(reset.ast)
#         self.expr_list = []
#         for reset in reset_list:
#             self.expr_list.append(reset.code)

#     def apply_reset_continuous(self, agent, continuous_variable_dict, lane_map):
#         agent_state_lower = []
#         agent_state_upper = []

#         # TODO-PARSER: Handle This
#         for var in agent.controller.vars_dict['ego'].cont:
#             agent_state_lower.append(continuous_variable_dict['ego.'+var][0])
#             agent_state_upper.append(continuous_variable_dict['ego.'+var][1])

#         # TODO-PARSER: Handle This
#         assert len(agent_state_lower) == len(agent_state_upper) == len(agent.controller.vars_dict['ego'].cont)
#         for expr in self.expr_list:
#             if 'mode' not in expr:
#                 tmp = expr.split('=')
#                 lhs, rhs = tmp[0], tmp[1]
#                 # TODO-PARSER: Handle This
#                 for lhs_idx, cts_variable in enumerate(agent.controller.vars_dict['ego'].cont):
#                     if "output."+cts_variable == lhs:
#                         break

#                 lower = float('inf')
#                 upper = -float('inf')

#                 symbols = []
#                 for var in continuous_variable_dict:
#                     if var in expr:
#                         symbols.append(var)

#                 combinations = self._get_combinations(symbols, continuous_variable_dict)
#                 # for cts_variable in continuous_variable_dict:
#                 #     tmp = tmp.replace(cts_variable, str(continuous_variable_dict[cts_variable]))
#                 # next_init[i] = eval(tmp)
#                 for i in combinations.shape[0]:
#                     comb = combinations[i,:]
#                     for j in range(len(symbols)):
#                         tmp = rhs.replace(symbols[j], str(comb[i,j]))
#                         tmp = min(tmp, lower)
#                         tmp = max(tmp, upper)

#                 agent_state_lower[lhs_idx] = lower
#                 agent_state_upper[lhs_idx] = upper

#         return [agent_state_lower, agent_state_upper]

#     def _get_combinations(self, symbols, cont_var_dict):
#         all_vars = []
#         for symbol in symbols:
#             all_vars.append(cont_var_dict[symbol])
#         comb_array = np.array(np.meshgrid(*all_vars)).T.reshape(-1, len(symbols))
#         return comb_array

#     def get_dest(self, agent, agent_state, discrete_variable_dict, lane_map) -> str:
#         agent_mode = agent_state[1]
#         dest = copy.deepcopy(agent_mode)
#         possible_dest = [[elem] for elem in dest]
#         for reset in self.expr_list:
#             if "mode" in reset:
#                 # TODO-PARSER: Handle This
#                 for i, discrete_variable_ego in enumerate(agent.controller.vars_dict['ego'].disc):
#                     if discrete_variable_ego in reset:
#                         break
#                 tmp = reset.split('=')
#                 if 'map' in tmp[1]:
#                     tmp = tmp[1]
#                     for var in discrete_variable_dict:
#                         tmp = tmp.replace(var, f"'{discrete_variable_dict[var]}'")
#                     res = eval(tmp)
#                     if not isinstance(res, list):
#                         res = [res]
#                     possible_dest[i] = res 
#                 else:
#                     tmp = tmp[1].split('.')
#                     # TODO-PARSER: Handle This
#                     if tmp[0].strip(' ') in agent.controller.modes:
#                         possible_dest[i] = [tmp[1]]
#         all_dest = itertools.product(*possible_dest)
#         res = []
#         for dest in all_dest:
#             res.append(dest)
#         return res
