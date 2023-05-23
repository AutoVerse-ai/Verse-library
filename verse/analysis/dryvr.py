import random, numpy as np
from typing import List, Tuple
from scipy import spatial

_TRUE_MIN_CONST = -10
_EPSILON = 1.0e-6
_SMALL_EPSILON = 1e-10
SIMTRACENUM = 10

PW = "PW"
GLOBAL = "GLOBAL"


def all_sensitivities_calc(training_traces: np.ndarray, initial_radii: np.ndarray):
    num_traces: int
    trace_len: int
    ndims: int
    num_traces, trace_len, ndims = training_traces.shape
    normalizing_initial_set_radii: np.array = initial_radii.copy()
    y_points: np.array = np.zeros((normalizing_initial_set_radii.shape[0], trace_len - 1))
    normalizing_initial_set_radii[np.where(normalizing_initial_set_radii == 0)] = 1.0
    for cur_dim_ind in range(1, ndims):
        # keyi: move out of loop
        normalized_initial_points: np.array = (
            training_traces[:, 0, 1:] / normalizing_initial_set_radii
        )
        initial_distances = (
            spatial.distance.pdist(normalized_initial_points, "chebyshev") + _SMALL_EPSILON
        )
        for cur_time_ind in range(1, trace_len):
            y_points[cur_dim_ind - 1, cur_time_ind - 1] = np.max(
                (
                    spatial.distance.pdist(
                        np.reshape(
                            training_traces[:, cur_time_ind, cur_dim_ind],
                            (training_traces.shape[0], 1),
                        ),
                        "chebychev",
                    )
                    / normalizing_initial_set_radii[cur_dim_ind - 1]
                )
                / initial_distances
            )
    return y_points


def get_reachtube_segment(
    training_traces: np.ndarray, initial_radii: np.ndarray, method="PWGlobal"
) -> np.array:
    num_traces: int = training_traces.shape[0]
    ndims: int = training_traces.shape[2]  # This includes time
    trace_len: int = training_traces.shape[1]
    center_trace: np.ndarray = training_traces[0, :, :]
    trace_initial_time = center_trace[0, 0]
    x_points: np.ndarray = center_trace[:, 0] - trace_initial_time
    assert np.all(training_traces[0, :, 0] == training_traces[1:, :, 0])
    y_points: np.ndarray = all_sensitivities_calc(training_traces, initial_radii)
    points: np.ndarray = np.zeros((ndims - 1, trace_len, 2))
    points[np.where(initial_radii != 0), 0, 1] = 1.0
    points[:, :, 0] = np.reshape(x_points, (1, x_points.shape[0]))
    points[:, 1:, 1] = y_points
    normalizing_initial_set_radii: np.ndarray = initial_radii.copy()
    normalizing_initial_set_radii[np.where(normalizing_initial_set_radii == 0)] = 1.0
    df: np.ndarray = np.zeros((trace_len, ndims))
    if method == "PW":
        df[:, 1:] = np.transpose(
            points[:, :, 1]
            * np.reshape(normalizing_initial_set_radii, (normalizing_initial_set_radii.size, 1))
        )
    elif method == "PWGlobal":
        # replace zeros with epsilons
        # points[np.where(points[:, 0, 1] == 0), 0, 1] = 1.0e-100
        # to fit exponentials make y axis log of sensitivity
        points[:, :, 1] = np.maximum(points[:, :, 1], _EPSILON)
        points[:, :, 1] = np.log(points[:, :, 1])
        for dim_ind in range(1, ndims):
            new_min = min(np.min(points[dim_ind - 1, 1:, 1]) + _TRUE_MIN_CONST, -10)
            if initial_radii[dim_ind - 1] == 0:
                # exclude initial set, then add true minimum points
                new_points: np.ndarray = np.row_stack(
                    (
                        np.array((points[dim_ind - 1, 1, 0], new_min)),
                        np.array((points[dim_ind - 1, -1, 0], new_min)),
                    )
                )
            else:
                # start from zero, then add true minimum points
                new_points: np.ndarray = np.row_stack(
                    (
                        points[dim_ind - 1, 0, :],
                        np.array((points[dim_ind - 1, 0, 0], new_min)),
                        np.array((points[dim_ind - 1, -1, 0], new_min)),
                    )
                )
                df[0, dim_ind] = initial_radii[dim_ind - 1]
                # Tuple order is start_time, end_time, slope, y-intercept
            cur_dim_points = np.concatenate((points[dim_ind - 1, 1:, :], new_points), axis=0)
            cur_hull: spatial.ConvexHull = spatial.ConvexHull(cur_dim_points)
            linear_separators: List[Tuple[float, float, float, float, int, int]] = []
            vert_inds = list(zip(cur_hull.vertices[:-1], cur_hull.vertices[1:]))
            vert_inds.append((cur_hull.vertices[-1], cur_hull.vertices[0]))
            for end_ind, start_ind in vert_inds:
                if (
                    cur_dim_points[start_ind, 1] != new_min
                    and cur_dim_points[end_ind, 1] != new_min
                ):
                    slope = (cur_dim_points[end_ind, 1] - cur_dim_points[start_ind, 1]) / (
                        cur_dim_points[end_ind, 0] - cur_dim_points[start_ind, 0]
                    )
                    y_intercept = (
                        cur_dim_points[start_ind, 1] - cur_dim_points[start_ind, 0] * slope
                    )
                    start_time = cur_dim_points[start_ind, 0]
                    end_time = cur_dim_points[end_ind, 0]
                    assert start_time < end_time
                    if start_time == 0:
                        linear_separators.append(
                            (start_time, end_time, slope, y_intercept, 0, end_ind + 1)
                        )
                    else:
                        linear_separators.append(
                            (start_time, end_time, slope, y_intercept, start_ind + 1, end_ind + 1)
                        )
            linear_separators.sort()
            prev_val = 0
            prev_ind = 1 if initial_radii[dim_ind - 1] == 0 else 0
            for linear_separator in linear_separators:
                _, _, slope, y_intercept, start_ind, end_ind = linear_separator
                assert prev_ind == start_ind
                assert start_ind < end_ind
                segment_t = center_trace[start_ind : end_ind + 1, 0]
                segment_df = (
                    normalizing_initial_set_radii[dim_ind - 1]
                    * np.exp(y_intercept)
                    * np.exp(slope * segment_t)
                )
                segment_df[0] = max(segment_df[0], prev_val)
                df[start_ind : end_ind + 1, dim_ind] = segment_df
                prev_val = segment_df[-1]
                prev_ind = end_ind
    else:
        print("Discrepancy computation method,", method, ", is not supported!")
        raise ValueError
    assert np.all(df >= 0)
    reachtube_segment: np.ndarray = np.zeros((trace_len - 1, 2, ndims))
    reachtube_segment[:, 0, :] = np.minimum(
        center_trace[1:, :] - df[1:, :], center_trace[:-1, :] - df[:-1, :]
    )
    reachtube_segment[:, 1, :] = np.maximum(
        center_trace[1:, :] + df[1:, :], center_trace[:-1, :] + df[:-1, :]
    )
    # assert 100% training accuracy (all trajectories are contained)
    for trace_ind in range(training_traces.shape[0]):
        if not (
            np.all(reachtube_segment[:, 0, :] <= training_traces[trace_ind, 1:, :])
            and np.all(reachtube_segment[:, 1, :] >= training_traces[trace_ind, 1:, :])
        ):
            assert np.any(
                np.abs(training_traces[trace_ind, 0, 1:] - center_trace[0, 1:]) > initial_radii
            )
            print(
                f"Warning: Trace #{trace_ind}",
                "of this initial set is sampled outside of the initial set because of floating point error and is not contained in the initial set",
            )
    return reachtube_segment


def calcCenterPoint(lower, upper):
    """
    Calculate the center point between the lower and upper bound
    The function only supports list since we assue initial set is always list

    Args:
        lower (list): lowerbound.
        upper (list): upperbound.

    Returns:
        delta (list of float)

    """

    # Convert list into float in case they are int
    lower = [float(val) for val in lower]
    upper = [float(val) for val in upper]
    assert len(lower) == len(upper), "Center Point List Range Error"
    return [(upper[i] + lower[i]) / 2 for i in range(len(upper))]


def calcDelta(lower, upper):
    """
    Calculate the delta value between the lower and upper bound
    The function only supports list since we assue initial set is always list

    Args:
        lower (list): lowerbound.
        upper (list): upperbound.

    Returns:
        delta (list of float)

    """
    # Convert list into float in case they are int
    lower = [float(val) for val in lower]
    upper = [float(val) for val in upper]

    assert len(lower) == len(upper), "Delta calc List Range Error"
    return [(upper[i] - lower[i]) / 2 for i in range(len(upper))]


def randomPoint(lower, upper, seed=None):
    """
    Pick a random point between lower and upper bound
    This function supports both int or list

    Args:
        lower (list or int or float): lower bound.
        upper (list or int or float): upper bound.

    Returns:
        random point (either float or list of float)

    """
    if seed is not None:
        random.seed(seed)

    if isinstance(lower, int) or isinstance(lower, float):
        return random.uniform(lower, upper)

    if isinstance(lower, list):
        assert len(lower) == len(upper), "Random Point List Range Error"

        return [random.uniform(lower[i], upper[i]) for i in range(len(lower))]


def trimTraces(traces):
    """
    trim all traces to the same length

    Args:
        traces (list): list of traces generated by simulator
    Returns:
        traces (list) after trim to the same length

    """

    trace_len = min(len(trace) for trace in traces)
    return [trace[:trace_len] for trace in traces]


def calc_bloated_tube(
    mode_label,
    initial_set,
    time_horizon,
    time_step,
    sim_func,
    bloating_method,
    kvalue,
    sim_trace_num,
    guard_checker=None,
    guard_str="",
    lane_map=None,
):
    """
    This function calculate the reach tube for single given mode

    Args:
        mode_label (str): mode name
        initial_set (list): a list contains upper and lower bound of the initial set
        time_horizon (float): time horizon to simulate
        sim_func (function): simulation function
        bloating_method (str): determine the bloating method for reach tube, either GLOBAL or PW
        sim_trace_num (int): number of simulations used to calculate the discrepancy
        kvalue (list): list of float used when bloating method set to PW
        guard_checker (verse.core.guard.Guard or None): guard check object
        guard_str (str): guard string

    Returns:
        Bloated reach tube

    """
    # print(initial_set)
    random.seed(4)
    cur_center = calcCenterPoint(initial_set[0], initial_set[1])
    cur_delta = calcDelta(initial_set[0], initial_set[1])
    traces = [sim_func(mode_label, cur_center, time_horizon, time_step, lane_map)]
    # Simulate SIMTRACENUM times to learn the sensitivity
    for i in range(sim_trace_num):
        new_init_point = randomPoint(initial_set[0], initial_set[1], i)
        traces.append(sim_func(mode_label, new_init_point, time_horizon, time_step, lane_map))

    # Trim the trace to the same length
    traces = trimTraces(traces)
    if guard_checker is not None:
        # pre truncated traces to get better bloat result
        max_idx = -1
        for trace in traces:
            ret_idx = guard_checker.guard_sim_trace_time(trace, guard_str)
            max_idx = max(max_idx, ret_idx + 1)
        for i in range(len(traces)):
            traces[i] = traces[i][:max_idx]

    # The major
    if bloating_method == GLOBAL:
        cur_reach_tube: np.ndarray = get_reachtube_segment(
            np.array(traces), np.array(cur_delta), "PWGlobal"
        )
        # cur_reach_tube: np.ndarray = ReachabilityEngine.get_reachtube_segment_wrapper(np.array(traces), np.array(cur_delta))
    elif bloating_method == PW:
        cur_reach_tube: np.ndarray = get_reachtube_segment(
            np.array(traces), np.array(cur_delta), "PW"
        )
        # cur_reach_tube: np.ndarray = ReachabilityEngine.get_reachtube_segment_wrapper(np.array(traces), np.array(cur_delta))
    else:
        raise ValueError("Unsupported bloating method '" + bloating_method + "'")
    final_tube = np.zeros((cur_reach_tube.shape[0] * 2, cur_reach_tube.shape[2]))
    final_tube[0::2, :] = cur_reach_tube[:, 0, :]
    final_tube[1::2, :] = cur_reach_tube[:, 1, :]
    # print(final_tube.tolist()[-2], final_tube.tolist()[-1])
    return final_tube
