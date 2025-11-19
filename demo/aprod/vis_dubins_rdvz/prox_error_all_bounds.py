import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Tuple

def wrap_angle(angle):
    """Wrap to [-pi, pi]."""
    # return np.arctan2(np.sin(angle), np.cos(angle))
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
        # Interval crosses the branch cut
        return angle >= ang_min or angle <= ang_max


def projection_bounds_general(theta, psi, eps_theta, axis):
    """
    Compute all candidate projections P for the given axis.
    axis ∈ {"x", "y", "z"}.
    """
    psi_min = wrap_angle(psi - eps_theta)
    psi_max = wrap_angle(psi + eps_theta)
    psi_candidates = [psi_min, psi_max]

    # Critical points for cos(psi) or sin(psi)
    if axis in ("x", "y"):
        # if psi_min <= 0 <= psi_max:
        if angle_in_wrapped_interval(0.0, psi_min, psi_max):
            psi_candidates.append(0)  # cos(0)=1
        # if psi_min <= np.pi <= psi_max:
        if angle_in_wrapped_interval(np.pi, psi_min, psi_max):
            psi_candidates.append(np.pi)  # cos(pi)=-1
    elif axis == "z":
        # if psi_min <= np.pi/2 <= psi_max:
        if angle_in_wrapped_interval(np.pi/2, psi_min, psi_max):
            psi_candidates.append(np.pi/2)  # sin(pi/2)=1
        # if psi_min <= -np.pi/2 <= psi_max:
        if angle_in_wrapped_interval(-np.pi/2, psi_min, psi_max):
            psi_candidates.append(-np.pi/2)  # sin(-pi/2)=-1

    # works since cos, sin are monotonically increasing on intervals that disclude critical points
    # Azimuth
    theta_min = wrap_angle(theta - eps_theta)
    theta_max = wrap_angle(theta + eps_theta)
    theta_candidates = [theta_min, theta_max]

    if axis in ("x", "y"):
        # if theta_min <= np.pi/2 <= theta_max:
        if angle_in_wrapped_interval(np.pi/2, theta_min, theta_max):
            theta_candidates.append(np.pi/2)
        # if theta_min <= -np.pi/2 <= theta_max:
        if angle_in_wrapped_interval(-np.pi/2, theta_min, theta_max):
            theta_candidates.append(-np.pi/2)
        if axis == "x":
            # if theta_min <= 0 <= theta_max:
            if angle_in_wrapped_interval(0, theta_min, theta_max):
                theta_candidates.append(0)
            # if theta_min <= np.pi <= theta_max:
            if angle_in_wrapped_interval(np.pi, theta_min, theta_max):
                theta_candidates.append(np.pi)
    else:
        # Z has no theta term
        theta_candidates = [0.0]

    # Combine
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
        (rho + eps_r)*P if P >= 0 else (rho - eps_r)*P # if P is positive, want positive rho error, else want smaller rho error to be less negative
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
    e_max, _ = point_error_general(xyz, eps_r, eps_theta, axis)
    return -e_max  # maximization → minimize negative

def error_min_obj(xyz, eps_r, eps_theta, axis):
    _, e_min = point_error_general(xyz, eps_r, eps_theta, axis)
    return e_min  # minimization: natural

def box_extreme_error(bounds, eps_r, eps_theta, axis) -> Tuple[float, float]:
    """
    returns: e-_max, e-_min
    bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    axis: "x", "y", or "z"
    To get standard 
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


# if __name__ == "__main__":
#     # Example usage for all 3 axes:
#     x, y, z = 2, 200, 2
#     eps_r = 0.05
#     eps_theta = np.deg2rad(2)

#     for axis in ["x", "y", "z"]:
#         # e_max, e_min = cartesian_error_bounds_at_point(x, y, z, eps_r, eps_theta, axis)
#         e_max, e_min = point_error_general((x, y, z), eps_r, eps_theta, axis)
#         print(f"{axis}-axis: Max +error: {e_max:.6f}, Max -error: {e_min:.6f}")

#     bounds = [(1.0, 2.0), (1.0, 200), (1.0, 2.0)]
#     print(f'Over bounds {bounds}')
#     for axis in ["x", "y", "z"]:
#         e_k_max, e_k_min = box_extreme_error(bounds, eps_r, eps_theta, axis)
#         print(f"{axis}-axis: Max +error: {e_k_max:.6f}, Max -error: {e_k_min:.6f}")

### angles-only error
def angular_bounds_rectangle(x_bounds, y_bounds):
    """
    Angles point from rectangle to origin
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

    # Case 1: rectangle contains origin
    if x_min < 0 < x_max and y_min < 0 < y_max:
        return -np.pi, np.pi

    # Case 2: rectangle does not cross -pi/pi
    if not (x_max < 0 and y_min < 0 < y_max):  # no 2nd/3rd quadrant wrap
        theta_min = angles.min()
        theta_max = angles.max()
        return theta_min, theta_max

    # Case 3: rectangle crosses -pi/pi (2nd/3rd quadrant)
    angles_mod = np.mod(angles, 2*np.pi)
    theta_min = angles_mod.min()
    theta_max = angles_mod.max()
    # map back to [-pi, pi]
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
    Expects arraylikes of from [xmin, xmax, ymin, ymax] -- can be obtained by doing np.array([xbounds, ybounds]).flatten \n
    Returns angular span of vectors pointing from rect1 to rect2 as [theta_min, theta_max]
    theta_max may be < theta_min in [-pi, pi] range if vectors spanned the pi/-pi wrapping point
    """
    x1min, x1max, y1min, y1max = rect1
    x2min, x2max, y2min, y2max = rect2

    # intersection → full circle
    if not (x1max < x2min or x2max < x1min or y1max < y2min or y2max < y1min):
        return -np.pi, np.pi

    c1 = rect_corners(x1min, x1max, y1min, y1max)
    c2 = rect_corners(x2min, x2max, y2min, y2max)

    vecs = (c2[:, None, :] - c1[None, :, :]).reshape(-1, 2) # using broadcasting tricks to compute all vectors
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    amin, amax = angles.min(), angles.max()

    if amax - amin <= np.pi:
        # no wrapping
        return amin, amax
    else:
        # wrapping: shift into [0, 2π), recompute bounds
        angles_mod = np.mod(angles, 2*np.pi)
        amin, amax = angles_mod.min(), angles_mod.max()
        # map back to [-π, π]
        if amin > np.pi:
            amin -= 2*np.pi
        if amax > np.pi:
            amax -= 2*np.pi
        return amin, amax

def angular_bounds_diff(theta, theta_ref) -> Tuple[float]: 
    """
    Assuming angles either span the entire interval [-pi,pi] or span at most pi
    theta_max<theta_min only in cases where theta crosses the pi/-pi wrapping point
    Returns the angular difference bound theta-theta_ref in the same format as above
    """
    theta_min, theta_max = theta 
    theta_ref_min, theta_ref_max = theta_ref
    if (theta_min == -np.pi and theta_max == np.pi) or (theta_ref_min == -np.pi and theta_ref_max == np.pi):
        return -np.pi, np.pi # if either interval is the entire circle, just return the circle
    theta_max = theta_max+2*np.pi if theta_max<theta_min else theta_max # wrap theta_max to [0,2pi] if < theta_min
    theta_ref_max = theta_ref_max+2*np.pi if theta_ref_max<theta_ref_min else theta_ref_max # likewise for theta_ref
    diff_min, diff_max = wrap_angle(theta_min-theta_ref_max), wrap_angle(theta_max-theta_ref_max)
    return diff_min, diff_max

if __name__ == "__main__":

    bounds = [(0.1, 1), (-1,0.1)]
    # bounds = [0.1, 1, -1,0.1]
    obs_bounds = [0,0, 0,0]
    print(f'Over bounds {bounds}')
    # min_arc, max_arc = find_angle_bounds(bounds[0], bounds[1])
    # min_arc, max_arc = angular_bounds_rectangle(bounds[0], bounds[1])
    min_arc, max_arc = angular_span_between_rects(np.array(bounds).flatten(), obs_bounds)
    print(f'Min angle {min_arc}, max angle: {max_arc}')
