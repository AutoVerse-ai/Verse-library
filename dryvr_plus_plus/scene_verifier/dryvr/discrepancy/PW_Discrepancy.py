from __future__ import division, print_function

import math


def read_data(traces):
    """ Read in all the traces """
    error_thred_time = 1e-3

    trace = traces[0]
    delta_time = trace[1][0] - trace[0][0]

    # Calculate variables
    dimensions = len(trace[0])
    dimensions_nt = dimensions - 1
    end_time = trace[-1][0]

    # Align all the traces
    for i in range(len(traces)):
        initial_time = traces[i][0][0]
        for j in range(len(traces[i])):
            traces[i][j][0] = traces[i][j][0] - initial_time

    # reassign the start time and end time
    start_time = 0
    for i in range(len(traces)):
        end_time = min(end_time, traces[i][-1][0])

    # trim all the points after the end time
    traces_trim = traces

    # reassign trace_len
    trace_len = len(traces_trim[0])

    return (traces_trim, dimensions, dimensions_nt, trace_len, end_time, delta_time, start_time)


def compute_diff(traces):
    """ Compute difference between traces """
    # Iterate over all combinations
    traces_diff = []
    for i in range(0, len(traces)):
        for j in range(i + 1, len(traces)):

            trace_diff = []

            # Iterate over the data of the trace
            for t in range(0, len(traces[i])):
                diff = [abs(x_i - y_i) for x_i, y_i in zip(traces[i][t],
                                                           traces[j][t])]
                trace_diff.append(diff[1:])

            # Append to traces diff minus time difference
            traces_diff.append(trace_diff)

    # TODO Eliminate hardcoded file name
    with open('output/tracediff2.txt', 'w') as write_file:
        for i in range(len(traces_diff)):
            for j in range(len(traces_diff[0])):
                write_file.write(str(traces_diff[i][j]) + '\n')
            write_file.write('\n')
    return traces_diff


def find_time_intervals(traces_diff, dimensions_nt, end_time, trace_len, delta_time, K_value):
    """ Compute the time intervals """
    # FIXME just do 1 dimension for now
    # Iterate through all dimensions
    num_ti = []
    time_intervals = []

    for i in range(0, dimensions_nt):

        time_dim = []

        # Obtain difference at start of interval
        diff_0 = []
        t_0 = 0.0
        time_dim.append(t_0)
        for k in range(0, len(traces_diff)):
            diff_0.append(traces_diff[k][0][i])
        # Iterate through all points in trace

        for j in range(1, trace_len):
            # Obtain difference at ith time of interval
            diff_i = []
            try:
                for k in range(0, len(traces_diff)):
                    diff_i.append(traces_diff[k][j][i])
            except IndexError:
                print(trace_len)
                print(k, j, i)
                print(len(traces_diff[k]))
                print(len(traces_diff[k][j]))

            # Check time
            t_i = j * delta_time
            t = t_i - t_0
            if t <= 0:
                continue

            # Compute ratios
            ratio = []
            for d_0, d_i in zip(diff_0, diff_i):
                if d_i < 1E-3:
                    continue
                elif d_0 < 1E-3:
                    continue

                # NOTE not sure if this is right?
                # ratio.append((1 / t) * math.log(d_i / d_0))
                ratio.append(d_i / d_0)

            # Check ratios if less than constant
            # new_int = all(r <= 2.0*K_value[i] for r in ratio)
            # new_int = all(r <= 2**(2*t)*K_value[i] for r in ratio)
            new_int = all(r <= 1 for r in ratio)
            if new_int == False:
                if t_i != end_time:
                    time_dim.append(t_i)
                diff_0 = diff_i
                t_0 = t_i

        # Append the time intervals
        time_dim.append(end_time)
        time_intervals.append(time_dim)
        # record the number of time intervals
        num_ti.append(len(time_intervals[i]) - 1)

    return (time_intervals, num_ti)


# Compute discrepancies
def calculate_discrepancies(time_intervals, traces_diff, dimensions_nt, delta_time, K_value):
    # FIXME
    # Iterate over all dimensions
    discrepancies = []
    for nd in range(0, dimensions_nt):
        # for nd in xrange(0, P_DIM):
        disc = []

        # Iterate over all time intervals
        for ni in range(0, len(time_intervals[nd]) - 1):
            t_0 = time_intervals[nd][ni]
            t_e = time_intervals[nd][ni + 1]
            # t_i = t_0 + delta_time

            # FIXME (???)
            # print "note",delta_time
            points = int((t_e - t_0) / delta_time + 0.5) + 1
            idx = int(t_0 / delta_time)

            # try to find the best K and gamma
            tmp_K_value = K_value[nd]
            # Iterate over all trace difference
            glpk_rows = []
            close_flag = 0
            for k in range(0, len(traces_diff)):

                # Compute initial
                diff_0 = traces_diff[k][0][nd]
                if diff_0 <= 1E-3:
                    # print('Find two traces to be too closed!')
                    # print('use the default value!')
                    close_flag = 1
                    break
                ln_0 = math.log(diff_0)

                # FIXME need to reset the delta_time here
                t_i = t_0 + delta_time
                # print(disc)
                # Obtain rows for GLPK
                for r in range(1, points):
                    t_d = t_i - t_0
                    t_i += delta_time
                    diff_i = traces_diff[k][idx + r][nd]

                    if diff_i < 1E-3:
                        continue

                    ln_i = math.log(diff_i)

                    # compute the existing previous time interval discrepancy
                    discrepancy_now = 0
                    if len(disc) != 0:
                        for time_prev in range(0, len(disc)):
                            discrepancy_now = discrepancy_now + disc[time_prev] * (
                                        time_intervals[nd][time_prev + 1] - time_intervals[nd][time_prev])

                    ln_d = ln_i - ln_0 - math.log(tmp_K_value) - discrepancy_now
                    glpk_rows.append([t_d, ln_d])

            # Debugging algebraic solution
            if close_flag == 0:
                alg = [d / t for t, d in glpk_rows]
                if len(alg) != 0:
                    alg_max = max(alg)
                else:
                    alg_max = 0
            else:
                alg_max = 0

            disc.append(alg_max)

        # Append discrepancies
        discrepancies.append(disc)

    return discrepancies


# Obtain bloated tube
def generate_bloat_tube(traces, time_intervals, discrepancies, Initial_Delta, end_time, trace_len, dimensions_nt,
                        delta_time, K_value):

    # Iterate over all dimensions
    # FIXME
    bloat_tube = []
    for i in range(trace_len):
        bloat_tube.append([])
        bloat_tube.append([])

    for nd in range(0, dimensions_nt):
        # for nd in xrange(P_DIM - 1, P_DIM):

        time_bloat = []
        low_bloat = []
        up_bloat = []

        # To construct the reach tube
        time_tube = []
        tube = []

        prev_delta = Initial_Delta[nd]

        # Iterate over all intervals
        previous_idx = -1

        for ni in range(0, len(time_intervals[nd]) - 1):
            t_0 = time_intervals[nd][ni]
            t_e = time_intervals[nd][ni + 1]

            if t_e == end_time:
                points = int((t_e - t_0) / delta_time + 0.5) + 1
            else:
                points = int((t_e - t_0) / delta_time + 0.5)
            idx = int(t_0 / delta_time)

            gamma = discrepancies[nd][ni]

            # Iterate over all points in center trace
            for r in range(0, points):

                current_idx = idx + r

                if current_idx != previous_idx + 1:
                    # print('Index mismatch found!')
                    if current_idx == previous_idx:
                        idx += 1
                    elif current_idx == previous_idx + 2:
                        idx -= 1

                pnt = traces[0][idx + r]
                pnt_time = pnt[0]
                pnt_data = pnt[nd + 1]

                cur_delta = prev_delta * math.exp(gamma * delta_time)
                max_delta = max(prev_delta, cur_delta)

                time_bloat.append(pnt_time)
                low_bloat.append(pnt_data - max_delta * K_value[nd])
                up_bloat.append(pnt_data + max_delta * K_value[nd])

                if nd == 0:
                    bloat_tube[2 * (idx + r)].append(pnt_time)
                    bloat_tube[2 * (idx + r)].append(pnt_data - max_delta * K_value[nd])
                    bloat_tube[2 * (idx + r) + 1].append(pnt_time + delta_time)
                    bloat_tube[2 * (idx + r) + 1].append(pnt_data + max_delta * K_value[nd])
                else:
                    bloat_tube[2 * (idx + r)].append(pnt_data - max_delta * K_value[nd])
                    bloat_tube[2 * (idx + r) + 1].append(pnt_data + max_delta * K_value[nd])

                prev_delta = cur_delta

                previous_idx = idx + r

    return bloat_tube

# Print out the intervals and discrepancies
def print_int_disc(discrepancies, time_intervals):
    for nd in range(0, len(discrepancies)):
        for p in range(0, len(discrepancies[nd])):
            print('idx: ' + str(p) + ' int: ' + str(time_intervals[nd][p])
                  + ' to ' + str(time_intervals[nd][p + 1]) + ', disc: ' +
                  str(discrepancies[nd][p]))
        print('')

def PW_Bloat_to_tube(Initial_Delta, plot_flag, plot_dim, traces, K_value):
    # Read data in
    # if Mode == 'Const':
    #     K_value = [1.0,1.0,2.0]
    # elif Mode == 'Brake':
    #     K_value = [1.0,1.0,7.0]

    # if Mode == 'Const;Const':
    #     K_value = [1.0,1.0,2.0,1.0,1.0,2.0]
    # elif Mode == 'Brake;Const':
    #     K_value = [1.0,1.0,2.0,1.0,1.0,2.0]

    # elif Mode == 'Brake;Brake':
    #     K_value = [1.0,1.0,5.0,1.0,1.0,2.0]

    traces, dimensions, dimensions_nt, trace_len, end_time, delta_time, start_time = read_data(traces)
    # Compute difference between traces
    traces_diff = compute_diff(traces)
    # print traces_diff

    # Find time intervals for discrepancy calculations
    time_intervals, num_ti = find_time_intervals(traces_diff, dimensions_nt, end_time, trace_len, delta_time, K_value)
    # print('number of time intervals:')
    # print num_ti
    # Discrepancy calculation
    discrepancies = calculate_discrepancies(time_intervals, traces_diff, dimensions_nt, delta_time, K_value)
    # print('The K values')
    # print K_value,
    # system.exit('test')

    # Write discrepancies to file
    # write_to_file(time_intervals,discrepancies,write_path +' disp.txt', 'disc')

    # Nicely print the intervals and discrepancies
    # print_int_disc(discrepancies,time_intervals)

    # Bloat the tube using time intervals
    reach_tube = generate_bloat_tube(traces, time_intervals, discrepancies, Initial_Delta, end_time, trace_len,
                                     dimensions_nt, delta_time, K_value)

    # if plot_flag:
    #     plot_traces(traces, plot_dim, reach_tube)
    #     plt.show()

    return reach_tube
