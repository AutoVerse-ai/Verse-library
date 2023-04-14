import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional, Union
from math import pi, cos, sin, acos, asin, atan, tan

from verse.analysis.utils import wrap_to_pi, Vector, get_class_path, class_from_path, to_serializable


class LineType_3d:

    """A lane side line type."""

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class AbstractLane_3d(object):

    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    longitudinal_start: float = 0
    line_types: List["LineType_3d"]

    def __init__(self, id: str):
        self.id = id
        self.type = None

    @abstractmethod
    def position(self, longitudinal: float, lateral: float, theta: float) -> np.ndarray:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def altitude(self):
        """
        Get the lane altitude (avg z of the start and end point)

        :return: the lane altitude [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: dict):
        """
        Create lane instance from config

        :param config: json dict with lane parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def to_config(self) -> dict:
        """
        Write lane parameters to dict which can be serialized to json

        :return: dict of lane parameters
        """
        raise NotImplementedError()

    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) \
            -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
            -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and \
            0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_close

    def after_end(self, position: np.ndarray, longitudinal: float = None, lateral: float = None) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position: np.ndarray):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(self, position: np.ndarray, heading: Optional[float], heading_weight: float = 1.0):
        """Compute a weighted distance in position and heading to the lane."""
        if heading is None:
            return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(wrap_to_pi(heading - self.heading_at(s)))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight*angle


class StraightLane_3d(AbstractLane_3d):

    """A lane going in straight line."""

    def __init__(self,
                 id: str,
                 start: Vector,
                 end: Vector,
                 width: float = AbstractLane_3d.DEFAULT_WIDTH,
                 line_types: Tuple[LineType_3d, LineType_3d] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for determining who has right of way
        """
        self.id = id
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        # self.heading = np.arctan2(
        #     self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [
            LineType_3d.STRIPED, LineType_3d.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit
        self.type = 'Straight'
        self.longitudinal_start = 0

    def position(self, longitudinal: float, lateral: float, theta: float) -> np.ndarray:
        point = get_coor_by_rt(
            lateral, theta, self.direction, self.start+longitudinal*self.direction)
        return point

    # def heading_at(self, longitudinal: float) -> float:
    #     return self.heading
    def altitude(self):
        return (self.start[2]+self.end[2])/2

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float, float]:
        delta = position - self.start
        n = self.direction
        cross = np.cross(n, delta)
        rget = np.linalg.norm(cross)
        lget = np.dot(delta, n)
        centerget = self.start+lget*n
        if round(rget, 3) == 0:
            return lget, rget, 0
        return float(lget), float(rget), get_theta_by_coor(position, n, centerget, rget, True)

    def get_tang(self, longitudinal: float):
        return self.direction

    def get_sample_points(self, num_theta, num_len):
        n = self.direction
        l1 = (n[0]**2+n[1]**2)**0.5
        if l1 != 0:
            l2 = 1.0
            l2 = (n[0]**2+n[1]**2+n[2]**2)**0.5
            a = np.array([n[1]/l1, -n[0]/l1, 0])
            b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
        else:
            a = np.array([1, 0, 0])
            b = np.array([0, 1, 0])
        n = self.direction
        start = self.start
        r = self.width
        thetas, ls = np.mgrid[0:2*pi:num_theta*1j, 0:self.length:num_len*1j]
        x = start[0]+ls*n[0]+r * \
            (a[0]*np.cos(thetas)+b[0]*np.sin(thetas))
        y = start[1]+ls*n[1]+r * \
            (a[1]*np.cos(thetas)+b[1]*np.sin(thetas))
        z = start[2]+ls*n[2]+r * \
            (a[2]*np.cos(thetas)+b[2]*np.sin(thetas))
        return x, y, z

    def get_lane_center(self, num):
        ls = np.mgrid[0:self.length:num*1j]
        nx = self.start[0]+self.direction[0]*ls
        ny = self.start[1]+self.direction[1]*ls
        nz = self.start[2]+self.direction[2]*ls
        oc = np.zeros((num, 3))
        oc[:, 0] = nx
        oc[:, 1] = ny
        oc[:, 2] = nz
        return oc, nx, ny, nz

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "start": to_serializable(self.start),
                "end": to_serializable(self.end),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority
            }
        }


class CircularLane_3d_v1(AbstractLane_3d):

    """A lane going in circle arc."""

    def __init__(self,
                 id,
                 center: Vector,
                 radius: float,
                 norm_vec: Vector,
                 start_phase: float,
                 end_phase: float,
                 right_rotate: bool = True,
                 width: float = AbstractLane_3d.DEFAULT_WIDTH,
                 line_types: List[LineType_3d] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__(id)
        self.center = np.array(center)
        self.radius = radius
        self.norm_vec = norm_vec/np.linalg.norm(norm_vec)
        n = self.norm_vec
        l1 = (n[0]**2+n[1]**2)**0.5
        if l1 != 0:
            l2 = (n[0]**2+n[1]**2+n[2]**2)**0.5
            self.a = np.array([n[1]/l1, -n[0]/l1, 0])
            self.b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
        else:
            self.a = np.array([1, 0, 0])
            if n[2] > 0:
                self.b = np.array([0, 1, 0])
            elif n[2] < 0:
                self.b = np.array([0, -1, 0])
            else:
                raise ValueError
        if not right_rotate:
            self.b = -self.b
        self.start_phase = start_phase
        self.end_phase = end_phase
        if right_rotate and self.end_phase <= self.start_phase:
            self.end_phase += 2*pi
        if not right_rotate and self.end_phase >= self.start_phase:
            self.start_phase += 2*pi
        if max(self.start_phase, self.end_phase) > 2*pi:
            self.start_phase -= 2*pi
            self.end_phase -= 2*pi
        # print(self.start_phase, self.end_phase)
        self.right_rotate = right_rotate
        self.direction = -1 if right_rotate else 1
        self.width = width
        self.line_types = line_types or [
            LineType_3d.STRIPED, LineType_3d.STRIPED]
        self.forbidden = forbidden
        self.length = abs(radius*(self.end_phase - self.start_phase))
        self.priority = priority
        self.speed_limit = speed_limit
        self.type = 'Circular'
        self.longitudinal_start = 0

    def position(self, longitudinal: float, lateral: float, theta: float) -> np.ndarray:
        outer_center = self.get_outer_center(longitudinal)
        cross = np.cross(self.norm_vec, outer_center-self.center)
        norm_cross = cross/np.linalg.norm(cross)
        point = get_coor_by_rt(lateral, theta, norm_cross, outer_center)
        return point

    # def heading_at(self, longitudinal: float) -> float:
    #     phi = self.direction * longitudinal / self.radius + self.start_phase
    #     psi = wrap_to_pi(phi + np.pi/2 * self.direction)
    #     return psi
    def altitude(self):
        start, start_norm, end, end_norm = self.get_start_end_tang()
        return (start[2]+end[2])/2

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float, float]:
        rget, thetaget, phaseget = get_rtp_by_coor(
            position, self.norm_vec, self.center, self.radius, self.right_rotate)
        if phaseget < self.start_phase and phaseget < self.end_phase:
            phaseget += 2*pi
        if phaseget > self.start_phase and phaseget > self.end_phase:
            phaseget -= 2*pi
        lget = (phaseget-self.start_phase)*self.length / \
            (self.end_phase - self.start_phase)
        return lget, rget, thetaget

    def get_outer_center(self, longitudinal: float):
        phase = self.start_phase + \
            (self.end_phase - self.start_phase)*longitudinal/self.length
        outer_center = get_coor_by_rt(
            self.radius, phase, self.norm_vec, self.center, self.right_rotate)
        return outer_center

    def get_tang(self, longitudinal: float):
        outer_center = self.get_outer_center(longitudinal)
        return np.cross(self.norm_vec, outer_center-self.center)

    def get_start_end_tang(self):
        start = self.get_outer_center(0)
        start_tang = np.cross(self.norm_vec, start-self.center)
        end = self.get_outer_center(self.length)
        end_tang = np.cross(self.norm_vec, end-self.center)
        # print('get_start_end_tang', start, start_tang, end, end_tang)
        return start, start_tang, end, end_tang

    def get_sample_points(self, num_theta, num_len):
        thetas = np.mgrid[0:2*pi:num_theta*1j]
        if (self.start_phase <= self.end_phase and self.right_rotate):
            phases = np.mgrid[self.start_phase:self.end_phase:num_len*1j]
        elif (self.start_phase <= self.end_phase and not self.right_rotate):
            phases = np.mgrid[self.end_phase:(
                self.start_phase+2*pi):num_len*1j]
        elif (self.start_phase >= self.end_phase and self.right_rotate):
            phases = np.mgrid[self.start_phase:(
                self.end_phase+2*pi):num_len*1j]
        elif (self.start_phase >= self.end_phase and not self.right_rotate):
            phases = np.mgrid[self.end_phase:self.start_phase:num_len*1j]

        n = self.norm_vec
        a = self.a
        b = self.b
        c_x, c_y, c_z = self.center
        r = self.radius
        oc_x = c_x+r * (a[0]*np.cos(phases)+b[0]*np.sin(phases))
        oc_y = c_y+r * (a[1]*np.cos(phases)+b[1]*np.sin(phases))
        oc_z = c_z+r * (a[2]*np.cos(phases)+b[2]*np.sin(phases))
        oc = np.zeros((num_len, 3))
        oc[:, 0] = oc_x
        oc[:, 1] = oc_y
        oc[:, 2] = oc_z
        n_out = np.cross(n, oc-self.center)

        vec_func_a = np.vectorize(func_a, signature='(n)->(n)')
        a_out = vec_func_a(n_out)
        vec_func_b = np.vectorize(func_b, signature='(n)->(n)')
        b_out = vec_func_b(n_out)
        r_out = self.width

        a_out_x = np.atleast_2d(a_out[:, 0])
        a_out_y = np.atleast_2d(a_out[:, 1])
        a_out_z = np.atleast_2d(a_out[:, 2])
        b_out_x = np.atleast_2d(b_out[:, 0])
        b_out_y = np.atleast_2d(b_out[:, 1])
        b_out_z = np.atleast_2d(b_out[:, 2])

        costheta = np.cos(thetas)[:, np.newaxis]
        sintheta = np.sin(thetas)[:, np.newaxis]

        oc_out_x = np.atleast_2d(
            oc_x)+(r_out * (a_out_x*costheta+b_out_x*sintheta))
        oc_out_y = np.atleast_2d(
            oc_y)+(r_out * (a_out_y*costheta+b_out_y*sintheta))
        oc_out_z = np.atleast_2d(
            oc_z)+(r_out * (a_out_z*costheta+b_out_z*sintheta))
        return oc_out_x, oc_out_y, oc_out_z

    def get_lane_center(self, num):
        if (self.start_phase <= self.end_phase and self.right_rotate):
            phases = np.mgrid[self.start_phase:self.end_phase:num*1j]
        elif (self.start_phase <= self.end_phase and not self.right_rotate):
            phases = np.mgrid[self.end_phase:(self.start_phase+2*pi):num*1j]
        elif (self.start_phase >= self.end_phase and self.right_rotate):
            phases = np.mgrid[self.start_phase:(self.end_phase+2*pi):num*1j]
        elif (self.start_phase >= self.end_phase and not self.right_rotate):
            phases = np.mgrid[self.end_phase:self.start_phase:num*1j]

        a = self.a
        b = self.b
        c_x, c_y, c_z = self.center
        r = self.radius
        oc_x = c_x+r * (a[0]*np.cos(phases)+b[0]*np.sin(phases))
        oc_y = c_y+r * (a[1]*np.cos(phases)+b[1]*np.sin(phases))
        oc_z = c_z+r * (a[2]*np.cos(phases)+b[2]*np.sin(phases))
        oc = np.zeros((num, 3))
        oc[:, 0] = oc_x
        oc[:, 1] = oc_y
        oc[:, 2] = oc_z
        return oc, oc_x, oc_y, oc_z

    @ classmethod
    def from_config(cls, config: dict):
        config["center"] = np.array(config["center"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "center": to_serializable(self.center),
                "radius": self.radius,
                "start_phase": self.start_phase,
                "end_phase": self.end_phase,
                "right_rotate": self.right_rotate,
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority
            }
        }


class CircularLane_3d_v2(CircularLane_3d_v1):
    def __init__(self,
                 id,
                 start: Vector,
                 end: Vector,
                 norm_vec: Vector,
                 delta_phase: float,
                 right_rotate: bool = True,
                 width: float = AbstractLane_3d.DEFAULT_WIDTH,
                 line_types: List[LineType_3d] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        center, radius = get_center_radius(
            start, end, norm_vec, delta_phase, right_rotate)
        start_phase = get_theta_by_coor(
            start, norm_vec, center, radius, right_rotate)
        end_phase = get_theta_by_coor(
            end, norm_vec, center, radius, right_rotate)
        # print(start_phase, end_phase)
        super().__init__(id, center, radius, norm_vec, start_phase, end_phase,
                         right_rotate, width, line_types, forbidden, speed_limit, priority)


def get_coor_by_rt(r, theta, n, center, right_rotate=True):
    l1 = (n[0]**2+n[1]**2)**0.5
    if l1 != 0:
        l2 = 1.0
        l2 = (n[0]**2+n[1]**2+n[2]**2)**0.5
        a = np.array([n[1]/l1, -n[0]/l1, 0])
        b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
        if not right_rotate:
            b = -b
        point = center+r*(a*cos(theta)+b*sin(theta))
    else:
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        if n[2] < 0:
            b = np.array([0, -1, 0])
        if not right_rotate:
            b = -b
        point = center+r*(a*cos(theta)+b*sin(theta))
    return point


def get_rtp_by_coor(point, n, center, radius, right_rotate=True):
    norm = np.cross(n, point-center)
    norm = norm/np.linalg.norm(norm)
    tang = np.cross(norm, n)
    tang = tang/np.linalg.norm(tang)*radius
    outer_center = center+tang
    rget = np.linalg.norm(point-outer_center)
    theta = get_theta_by_coor(point, norm, outer_center, rget, right_rotate)
    phase = get_theta_by_coor(outer_center, n, center, radius, right_rotate)
    return rget, theta, phase


def get_theta_by_coor(point, n, center, r, right_rotate=True):
    if round(r, 3) == 0:
        return 0
    delta2 = (point-center)/r
    num_digit1 = 5
    l1 = (n[0]**2+n[1]**2)**0.5
    if l1 != 0:
        l2 = 1.0
        a = np.array([n[1]/l1, -n[0]/l1, 0])
        b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
        if not right_rotate:
            b = -b
        sinx = round(delta2[2]/b[2], num_digit1)
        if n[0] != 0:
            cosx = round((delta2[1]-b[1]*sinx)/a[1], num_digit1)
        if n[1] != 0:
            cosx = round((delta2[0]-b[0]*sinx)/a[0], num_digit1)
    else:
        sinx = round(delta2[1], num_digit1)
        if n[2] < 0:
            sinx = -sinx
        elif n[2] == 0:
            raise ValueError
        if not right_rotate:
            sinx = -sinx
        cosx = round(delta2[0], num_digit1)
    if round(cosx, 5) == 0:
        if sinx > 0:
            theta = pi/2
        elif sinx < 0:
            theta = 1.5*pi
        else:
            theta = 0
    else:
        theta = atan(sinx/cosx)
        if cosx < 0:
            theta = pi+theta
        elif theta < 0 and cosx > 0:
            theta = 2*pi+theta
    return theta


def func_a(n):
    l1 = (n[0]**2+n[1]**2)**0.5
    if l1 != 0:
        a = np.array([n[1]/l1, -n[0]/l1, 0])
    else:
        a = np.array([1, 0, 0])
    return a


def func_b(n):
    l1 = (n[0]**2+n[1]**2)**0.5
    if l1 != 0:
        l2 = (n[0]**2+n[1]**2+n[2]**2)**0.5
        b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
    else:
        b = np.array([0, 1, 0])
    return b


def get_center_radius(start, end, n, phase, right_rotate=True):
    n, end, start = np.array(n), np.array(end), np.array(start)
    assert (2*pi > phase > 0 and np.any(start != end))
    if right_rotate:
        l_n = np.cross(n, start-end)
    else:
        l_n = np.cross(n, end-start)
    l_n = l_n/np.linalg.norm(l_n)
    mid = (end+start)/2
    l = np.linalg.norm(end-start)/2
    if phase > pi:
        theta = (phase-pi)/2
    elif phase < pi:
        theta = phase/2
    else:
        return mid, l/2
    return mid+l/tan(theta)*l_n, l/sin(theta)
