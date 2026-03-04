import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Tuple

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    if angle % (2 * np.pi) == np.pi:
        return np.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angle_in_wrapped_interval(angle, ang_min, ang_max):
    """
    Returns True if 'angle' ∈ [ang_min, ang_max] with wrap-around.
    All angles should be in [-π, π).
    """
    angle = wrap_angle(angle)
    ang_min = wrap_angle(ang_min)
    ang_max = wrap_angle(ang_max)

    if ang_min <= ang_max:
        return ang_min <= angle <= ang_max
    else:
        return angle >= ang_min or angle <= ang_max


def projection_bounds_general(theta, psi, eps_theta, axis):
    """
    Compute all candidate projections P for the given axis.
    axis ∈ {"x", "y", "z"}.
    """
    psi_min = wrap_angle(psi - eps_theta)
    psi_max = wrap_angle(psi + eps_theta)
    psi_candidates = [psi_min, psi_max]

    if axis in ("x", "y"):
        if angle_in_wrapped_interval(0.0, psi_min, psi_max):
            psi_candidates.append(0)
        if angle_in_wrapped_interval(np.pi, psi_min, psi_max):
            psi_candidates.append(np.pi)
    elif axis == "z":
        if angle_in_wrapped_interval(np.pi/2, psi_min, psi_max):
            psi_candidates.append(np.pi/2)
        if angle_in_wrapped_interval(-np.pi/2, psi_min, psi_max):
            psi_candidates.append(-np.pi/2)

    theta_min = wrap_angle(theta - eps_theta)
    theta_max = wrap_angle(theta + eps_theta)
    theta_candidates = [theta_min, theta_max]

    if axis in ("x", "y"):
        if angle_in_wrapped_interval(np.pi/2, theta_min, theta_max):
            theta_candidates.append(np.pi/2)
        if angle_in_wrapped_interval(-np.pi/2, theta_min, theta_max):
            theta_candidates.append(-np.pi/2)
        if axis == "x":
            if angle_in_wrapped_interval(0, theta_min, theta_max):
                theta_candidates.append(0)
            if angle_in_wrapped_interval(np.pi, theta_min, theta_max):
                theta_candidates.append(np.pi)
    else:
        theta_candidates = [0.0]

    P_values = []
    for psi_val in psi_candidates:
        cos_psi = np.cos(psi_val)
        sin_psi = np.sin(psi_val)
        for theta_val in theta_candidates:
            cos_theta = np.cos(theta_val)
            sin_theta = np.sin(theta_val)
            if axis == "x":
                P = cos_psi * cos_theta
            elif axis == "y":
                P = cos_psi * sin_theta
            elif axis == "z":
                P = sin_psi
            else:
                raise ValueError("Axis must be 'x', 'y', or 'z'")
            P_values.append(P)

    return np.array(P_values)

def point_error_general(xyz, eps_r, eps_theta, axis):
    x, y, z = xyz
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    psi = np.arctan2(z, np.sqrt(x**2 + y**2))

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if axis == "x":
        P_true = cos_psi * cos_theta
    elif axis == "y":
        P_true = cos_psi * sin_theta
    elif axis == "z":
        P_true = sin_psi

    P_vals = projection_bounds_general(theta, psi, eps_theta, axis)

    sensed_max = max(
        (rho + eps_r)*P if P >= 0 else (rho - eps_r)*P
        for P in P_vals
    )
    sensed_min = min(
        (rho - eps_r)*P if P >= 0 else (rho + eps_r)*P
        for P in P_vals
    )

    e_max = sensed_max - rho * P_true
    e_min = sensed_min - rho * P_true

    return e_max, e_min

# --- Outer box optimizer ---

def error_max_obj(xyz, eps_r, eps_theta, axis):
    """Objective for maximizing error (negated for minimization)."""
    e_max, _ = point_error_general(xyz, eps_r, eps_theta, axis)
    return -e_max

def error_min_obj(xyz, eps_r, eps_theta, axis):
    """Objective for minimizing error."""
    _, e_min = point_error_general(xyz, eps_r, eps_theta, axis)
    return e_min

def box_extreme_error(bounds, eps_r, eps_theta, axis) -> Tuple[float, float]:
    """
    Compute extreme error bounds using differential evolution optimization.
    
    Returns: e_max, e_min
    Args:
        bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        axis: "x", "y", or "z"
    """
    res_max = differential_evolution(
        error_max_obj, bounds=bounds, args=(eps_r, eps_theta, axis),
        updating="deferred", polish=True
    )
    e_k_max = -res_max.fun

    res_min = differential_evolution(
        error_min_obj, bounds=bounds, args=(eps_r, eps_theta, axis),
        updating="deferred", polish=True
    )
    e_k_min = res_min.fun

    return e_k_max, e_k_min

def angular_bounds_rectangle(x_bounds, y_bounds):
    """
    Compute angular bounds for vectors pointing from rectangle to origin.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x_min, x_max = -x_max, -x_min
    y_min, y_max = -y_max, -y_min
    
    corners = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_min),
        (x_max, y_max)
    ]
    angles = np.array([np.arctan2(y, x) for x, y in corners])

    if x_min < 0 < x_max and y_min < 0 < y_max:
        return -np.pi, np.pi

    if not (x_max < 0 and y_min < 0 < y_max):
        theta_min = angles.min()
        theta_max = angles.max()
        return theta_min, theta_max

    angles_mod = np.mod(angles, 2*np.pi)
    theta_min = angles_mod.min()
    theta_max = angles_mod.max()
    theta_min = (theta_min + np.pi) % (2*np.pi) - np.pi
    theta_max = (theta_max + np.pi) % (2*np.pi) - np.pi
    return theta_min, theta_max

def rect_corners(xmin, xmax, ymin, ymax):
    """Return the 4 corners of an axis-aligned rectangle."""
    return np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax]
    ])

def angular_span_between_rects(rect1, rect2):
    """
    Returns angular span of vectors pointing from rect1 to rect2.
    
    Args:
        rect1, rect2: arraylikes of form [xmin, xmax, ymin, ymax]
    
    Returns:
        (theta_min, theta_max) where theta_max may be < theta_min if wrapping occurs
    """
    x1min, x1max, y1min, y1max = rect1
    x2min, x2max, y2min, y2max = rect2

    if (x1min < x2min and x2min < x1max and y1min < y2min and y2min < y1max) or (x2min < x1min and x1min < x2max and y2min < y1min and y1min < y2max):
        return -np.pi, np.pi

    c1 = rect_corners(x1min, x1max, y1min, y1max)
    c2 = rect_corners(x2min, x2max, y2min, y2max)

    vecs = (c2[:, None, :] - c1[None, :, :]).reshape(-1, 2)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    amin, amax = angles.min(), angles.max()

    if amax - amin <= np.pi:
        return amin, amax
    else:
        angles_mod = np.mod(angles, 2*np.pi)
        amin, amax = angles_mod.min(), angles_mod.max()
        if amin > np.pi:
            amin -= 2*np.pi
        if amax > np.pi:
            amax -= 2*np.pi
        return amin, amax
    
def angular_span_rect(rect, split: bool = False):
    """
    Essentially atan2 for an entire rectangle.
    
    Args:
        rect: arraylike of form [xmin, xmax, ymin, ymax]
        split: whether to split wrapped intervals
    """
    theta_min, theta_max =  angular_span_between_rects([0, 0, 0, 0], rect)
    if not split or theta_min < theta_max:
        return [(theta_min, theta_max)]
    else:
        return [(theta_min, np.pi), (-np.pi, theta_max)]

def angular_span_rect_parser(y_bounds, x_bounds):
    """
    Essentially atan2 for an entire rectangle.
    
    Args:
        y_bounds, x_bounds: tuples of (min, max)
    """
    rect = list(x_bounds) + list(y_bounds)
    theta_min, theta_max =  angular_span_between_rects([0, 0, 0, 0], rect)
    if theta_min <= theta_max:
        return (theta_min, theta_max)
    else:
        return [(theta_min, np.pi), (-np.pi, theta_max)]


def angular_bounds_diff_correct(theta, theta_ref) -> Tuple[float]:
    """
    Returns the bounds for angular difference theta-theta_ref with the range [-pi, pi].
    
    Args:
        theta: angular bound with theta_min >= -pi, theta_max <= pi
        theta_ref: angular reference bound with theta_min >= -pi, theta_max <= pi
    """
    theta_min, theta_max = theta 
    theta_ref_min, theta_ref_max = theta_ref
    if (theta_min == -np.pi and theta_max == np.pi) or (theta_ref_min == -np.pi and theta_ref_max == np.pi):
        return -np.pi, np.pi
    
    diff_min_raw = theta_min - theta_ref_max
    diff_max_raw = theta_max - theta_ref_min
    
    n = 0
    diff_min = diff_min_raw
    while diff_min < 0:
        diff_min += 2*np.pi
        n += 1
    
    diff_max = diff_max_raw + 2*np.pi*n
    
    if diff_min <= np.pi <= diff_max:
        return (diff_min, np.pi), (-np.pi, diff_max - 2*np.pi)
    else:
        diff_min_wrapped = wrap_angle(diff_min)
        diff_max_wrapped = wrap_angle(diff_max)
        
        if diff_min_wrapped > diff_max_wrapped:
            return (diff_min_wrapped, np.pi), (-np.pi, diff_max_wrapped)
        else:
            return diff_min_wrapped, diff_max_wrapped

def combine_angular_bounds(bounds_list):
    """
    Combine a list of angular bounds without adding conservativeness.
    Each bound is either a single interval (min, max) or a tuple of two intervals
    representing wrapped intervals like [(a, π), (-π, b)].
    
    Args:
        bounds_list: List of bounds, each being either:
                    - (min, max): single interval
                    - ((min1, max1), (min2, max2)): wrapped interval (two parts)
    
    Returns:
        Combined bounds in the same format, merged where possible

    Note:
        Current implementation is suboptimal w.r.t. both sorting, why sort both negative and positive, and list merging. 
    """
    
    if not bounds_list:
        return []
    
    normalized = []
    for bound in bounds_list:
        if isinstance(bound[0], tuple):
            normalized.extend(list(bound))
        else:
            min_val, max_val = bound
            if min_val > max_val:
                normalized.append((min_val, np.pi))
                normalized.append((-np.pi, max_val))
            else:
                normalized.append((min_val, max_val))
    
    if not normalized:
        return []
    
    def sort_key(interval):
        start, end = interval
        if start >= 0:
            return (0, start)
        else:
            return (1, start)
    
    normalized.sort(key=sort_key)
    
    merged = [normalized[0]]
    
    for current in normalized[1:]:
        last = merged[-1]
        last_start, last_end = last
        curr_start, curr_end = current
        
        if _intervals_overlap_or_adjacent(last, current):
            new_start = min(last_start, curr_start)
            new_end = max(last_end, curr_end)
            merged[-1] = (new_start, new_end)
        else:
            merged.append(current)
    
    if len(merged) > 1:
        merged = _intervals_overlap_or_adjacent_wrapped(merged)
    
    return _format_bounds(merged)


def _intervals_overlap_or_adjacent(interval1, interval2):
    """
    Check if two intervals in [-π, π] overlap or are adjacent.
    Assumes both are non-wrapping intervals and interval1.start < interval2.start.
    """
    s1, e1 = interval1
    s2, e2 = interval2
    
    if s1 <= e1 and s2 <= e2:
        return s1 <= e2 and s2 <= e1
    
    return False


def _intervals_overlap_or_adjacent_wrapped(merged_intervals):
    """
    After initial merging, check if the last negative interval wraps into
    positive territory and merge accordingly.
    """
    if len(merged_intervals) < 2:
        return merged_intervals
    
    last_positive_idx = -1
    for i in range(len(merged_intervals)):
        if merged_intervals[i][0] >= 0:
            last_positive_idx = i
    
    if last_positive_idx == -1 or last_positive_idx == len(merged_intervals) - 1:
        return merged_intervals
    
    last_neg_start, last_neg_end = merged_intervals[-1]
    
    if last_neg_end <= 0:
        return merged_intervals
    
    largest_ei_idx = -1
    for i in range(last_positive_idx + 1):
        if last_neg_end > merged_intervals[i][1]:
            largest_ei_idx = i
    
    if largest_ei_idx == -1:
        return merged_intervals
    
    merged_intervals = merged_intervals[:last_positive_idx - largest_ei_idx] + [(last_neg_start, merged_intervals[largest_ei_idx][1])]
    
    return merged_intervals


def _format_bounds(merged):
    """
    Convert merged intervals back to the original format.
    Returns either single tuples or wrapped tuples.
    """
    if not merged:
        return []
    
    positive_parts = [iv for iv in merged if iv[0] >= 0]
    negative_parts = [iv for iv in merged if iv[1] <= 0]
    
    result = []
    
    if positive_parts and negative_parts:
        pos = positive_parts[0]
        neg = negative_parts[0]
        return (pos, neg)
    elif positive_parts:
        return [iv for iv in merged if iv[0] >= 0]
    elif negative_parts:
        return [iv for iv in merged if iv[1] <= 0]
    else:
        return merged
    
if __name__ == "__main__":
    theta = [0.1, np.pi]
    theta_ref = [0, np.pi/2]
    print(f'Diff bounds: {angular_bounds_diff_correct(theta, theta_ref)}')

""" 
NOTE: The wrap-around logic in _intervals_overlap_or_adjacent_wrapped handles edge cases
where intervals cross the -π/π boundary. Test thoroughly when modifying merge behavior.
"""