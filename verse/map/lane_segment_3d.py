import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional, Union
from math import pi, cos, sin, acos, asin

from verse.analysis.utils import wrap_to_pi, Vector, get_class_path, class_from_path, to_serializable
from verse.map.lane_segment import LineType, AbstractLane, StraightLane


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
    line_types: List["LineType"]

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
                 line_types: Tuple[LineType, LineType] = None,
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
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array(
            [-self.direction[1], self.direction[0], 0])
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
        return float(lget), float(rget), get_theta_by_coor(position, n, centerget, rget)

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


class CircularLane_3d(AbstractLane_3d):

    """A lane going in circle arc."""

    def __init__(self,
                 id,
                 center: Vector,
                 radius: float,
                 norm_vec: Vector,
                 start_phase: float,
                 end_phase: float,
                 clockwise: bool = True,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: List[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__(id)
        self.center = np.array(center)
        self.radius = radius
        self.norm_vec = norm_vec/np.linalg.norm(norm_vec)
        self.start_phase = start_phase
        self.end_phase = end_phase
        self.clockwise = clockwise
        self.direction = -1 if clockwise else 1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.length = abs(radius*(end_phase - start_phase))
        self.priority = priority
        self.speed_limit = speed_limit
        self.type = 'Circular'
        self.longitudinal_start = 0

    def position(self, longitudinal: float, lateral: float, theta: float) -> np.ndarray:
        phase = self.start_phase + \
            (self.end_phase - self.start_phase)*longitudinal/self.length
        outer_center = get_coor_by_rt(
            self.radius, phase, self.norm_vec, self.center)
        cross = np.cross(self.norm_vec, outer_center-self.center)
        norm_cross = cross/np.linalg.norm(cross)
        point = get_coor_by_rt(lateral, theta, norm_cross, outer_center)
        return point

    # def heading_at(self, longitudinal: float) -> float:
    #     phi = self.direction * longitudinal / self.radius + self.start_phase
    #     psi = wrap_to_pi(phi + np.pi/2 * self.direction)
    #     return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float, float]:
        rget, thetaget, phaseget = get_rtp_by_coor(
            position, self.norm_vec, self.center, self.radius)
        lget = (phaseget-self.start_phase)*self.length / \
            (self.end_phase - self.start_phase)
        return lget, rget, thetaget

    @classmethod
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
                "clockwise": self.clockwise,
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority
            }
        }


def get_coor_by_rt(r, theta, n, center):
    l1 = (n[0]**2+n[1]**2)**0.5
    if l1 != 0:
        l2 = 1.0
        l2 = (n[0]**2+n[1]**2+n[2]**2)**0.5
        a = np.array([n[1]/l1, -n[0]/l1, 0])
        b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
        point = center+r*(a*cos(theta)+b*sin(theta))
    else:
        point = center+r*np.array([cos(theta), sin(theta), 0])
    # print(point)
    return point


def get_rtp_by_coor(point, n, center, radius):
    norm = np.cross(n, point-center)
    norm = norm/np.linalg.norm(norm)
    tang = np.cross(norm, n)
    tang = tang/np.linalg.norm(tang)*radius
    outer_center = center+tang
    # print(outer_center)
    rget = np.linalg.norm(point-outer_center)
    theta = get_theta_by_coor(point, norm, outer_center, rget)
    # print(theta)
    phase = get_theta_by_coor(outer_center, n, center, radius)
    # print(phase)
    return rget, theta, phase


def get_theta_by_coor(point, n, center, r):
    delta2 = (point-center)/r
    l1 = (n[0]**2+n[1]**2)**0.5
    if l1 != 0:
        l2 = 1.0
        a = np.array([n[1]/l1, -n[0]/l1, 0])
        b = np.array([n[0]*n[2]/l1/l2, n[1]*n[2]/l1/l2, -l1/l2])
        sinx = round(delta2[2]/b[2], 3)
        if n[0] != 0:
            cosx = round((delta2[1]-b[1]*sinx)/a[1], 3)
        if n[1] != 0:
            cosx = round((delta2[0]-b[0]*sinx)/a[0], 3)
    else:
        sinx = round(delta2[1], 3)
        cosx = round(delta2[0], 3)
    theta1 = asin(sinx)
    theta2 = acos(cosx)
    if sinx >= 0 and cosx >= 0:
        pass
    elif sinx >= 0 and cosx < 0:
        theta1 = pi-theta1
    elif sinx < 0 and cosx >= 0:
        theta1 = 2*pi+theta1
        theta2 = 2*pi-theta2
    elif sinx < 0 and cosx < 0:
        theta1 = pi-theta1
        theta2 = 2*pi-theta2
    # print(theta1, theta2)
    if round(theta1, 3) == round(theta2, 3):
        return round(theta1, 3)
    else:
        raise ValueError("error theta")
