from typing import Tuple, List

import numpy as np
from scipy.integrate import ode

from verse import BaseAgent
from verse import LaneMap
from waypoint_autopilot import WaypointAutopilot
from scipy.integrate import RK45

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler


class F16_Agent(BaseAgent):
    '''Dynamics of an F16 aircraft
    derived from Stanley Bak's python library'''

    def __init__(self, id, ap: WaypointAutopilot, extended_states=False, integrator_str='rk45', model_str='morelli', v2_integrators=False, code=None, file_name=None):
        '''Contructor for one F16 agent
            EXACTLY one of the following should be given
            file_name: name of the controller
            code: pyhton string ddefning the controller
            '''
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)
        self.ap = ap
        self.extended_states = extended_states
        self.integrator_str = integrator_str
        self.model_str = model_str
        self.v2_integrators = v2_integrators

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:
        initial_state = np.array(initialCondition, dtype=float)
        llc = self.ap.llc

        num_vars = len(get_state_names()) + llc.get_num_integrators()

        if initial_state.size < num_vars:
            # append integral error states to state vector
            x0 = np.zeros(num_vars)
            x0[:initial_state.shape[0]] = initial_state
        else:
            x0 = initial_state

        assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"

        # run the numerical simulation
        times = [0]
        states = [x0]

        # mode can change at time 0
        self.ap.advance_discrete_mode(times[-1], states[-1])

        modes = [self.ap.mode]

        if self.extended_states:
            xd, u, Nz, ps, Ny_r = get_extended_states(
                self.ap, times[-1], states[-1], self.model_str, self.v2_integrators)

            xd_list = [xd]
            u_list = [u]
            Nz_list = [Nz]
            ps_list = [ps]
            Ny_r_list = [Ny_r]

        der_func = make_der_func(self.ap, self.model_str, self.v2_integrators)

        if self.integrator_str == 'rk45':
            integrator_class = RK45
            kwargs = {}
        else:
            assert self.integrator_str == 'euler'
            integrator_class = Euler
            kwargs = {'step': time_step}

        # note: fixed_step argument is unused by rk45, used with euler
        integrator = integrator_class(
            der_func, times[-1], states[-1], time_bound, **kwargs)

        while integrator.status == 'running':
            integrator.step()

            if integrator.t >= times[-1] + time_step:
                dense_output = integrator.dense_output()

                while integrator.t >= times[-1] + time_step:
                    t = times[-1] + time_step
                    #print(f"{round(t, 2)} / {tmax}")

                    times.append(t)
                    states.append(dense_output(t))

                    updated = self.ap.advance_discrete_mode(
                        times[-1], states[-1])
                    modes.append(self.ap.mode)

                    # re-run dynamics function at current state to get non-state variables
                    if self.extended_states:
                        xd, u, Nz, ps, Ny_r = get_extended_states(
                            self.ap, times[-1], states[-1], self.model_str, self.v2_integrators)

                        xd_list.append(xd)
                        u_list.append(u)

                        Nz_list.append(Nz)
                        ps_list.append(ps)
                        Ny_r_list.append(Ny_r)

                    if self.ap.is_finished(times[-1], states[-1]):
                        # this both causes the outer loop to exit and sets res['status'] appropriately
                        integrator.status = 'autopilot finished'
                        break

                    if updated:
                        # re-initialize the integration class on discrete mode switches
                        integrator = integrator_class(
                            der_func, times[-1], states[-1], time_bound, **kwargs)
                        break

        assert 'finished' in integrator.status

        res = {}
        res['status'] = integrator.status
        res['times'] = times
        res['states'] = np.array(states, dtype=float)
        res['modes'] = modes

        if self.extended_states:
            res['xd_list'] = xd_list
            res['ps_list'] = ps_list
            res['Nz_list'] = Nz_list
            res['Ny_r_list'] = Ny_r_list
            res['u_list'] = u_list
        trace = []
        state_len = len(get_state_names())
        for i in range(0, len(times)):
            trace.append([times[i]]+states[i]
                         [:state_len].tolist())
        return np.array(trace)


def make_der_func(ap: WaypointAutopilot, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]
            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc,
                                model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func


def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(
            t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r
