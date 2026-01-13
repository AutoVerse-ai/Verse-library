from scipy.optimize import differential_evolution

# NOTE: this should work fairly well in practice given that the bounds are not too large, but unclear if these are actually guaranteed to be overapproximations or good approximations
def get_heading_bounds_optimized(lane_map, lane_idx, x_bounds, y_bounds):
    """
    More precise bound computation using optimization
    """
    def neg_heading(pos):
        return -lane_map.get_lane_heading(lane_idx, pos)
    
    def pos_heading(pos):
        return lane_map.get_lane_heading(lane_idx, pos)
    
    bounds = [(x_bounds[0], x_bounds[1]), (y_bounds[0], y_bounds[1])]

    res_min = differential_evolution(pos_heading, bounds)
    res_max = differential_evolution(neg_heading, bounds)

    return [res_min.fun, -res_max.fun]

def get_lateral_distance_bounds_optimized(lane_map, lane_idx, x_bounds, y_bounds):
    """
    More precise bound computation using optimization
    """
    def neg_distance(pos):
        return -lane_map.get_lateral_distance(lane_idx, pos)
    
    def pos_distance(pos):
        return lane_map.get_lateral_distance(lane_idx, pos)
    
    bounds = [(x_bounds[0], x_bounds[1]), (y_bounds[0], y_bounds[1])]

    res_min = differential_evolution(pos_distance, bounds)
    res_max = differential_evolution(neg_distance, bounds)

    return [res_min.fun, -res_max.fun]

def get_long_pos_bounds_optimized(lane_map, lane_idx, x_bounds, y_bounds):
    """
    More precise bound computation using optimization
    """
    def neg_pos(pos):
        return -lane_map.get_longitudinal_position(lane_idx, pos)
    
    def pos_pos(pos):
        return lane_map.get_longitudinal_position(lane_idx, pos)
    
    bounds = [(x_bounds[0], x_bounds[1]), (y_bounds[0], y_bounds[1])]

    res_min = differential_evolution(pos_pos, bounds)
    res_max = differential_evolution(neg_pos, bounds)

    return [res_min.fun, -res_max.fun]