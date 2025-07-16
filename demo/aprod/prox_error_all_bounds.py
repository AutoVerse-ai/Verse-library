import numpy as np
from scipy.optimize import differential_evolution
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


if __name__ == "__main__":
    # Example usage for all 3 axes:
    x, y, z = 2, 200, 2
    eps_r = 0.05
    eps_theta = np.deg2rad(2)

    for axis in ["x", "y", "z"]:
        # e_max, e_min = cartesian_error_bounds_at_point(x, y, z, eps_r, eps_theta, axis)
        e_max, e_min = point_error_general((x, y, z), eps_r, eps_theta, axis)
        print(f"{axis}-axis: Max +error: {e_max:.6f}, Max -error: {e_min:.6f}")

    bounds = [(1.0, 2.0), (1.0, 200), (1.0, 2.0)]
    print(f'Over bounds {bounds}')
    for axis in ["x", "y", "z"]:
        e_k_max, e_k_min = box_extreme_error(bounds, eps_r, eps_theta, axis)
        print(f"{axis}-axis: Max +error: {e_k_max:.6f}, Max -error: {e_k_min:.6f}")