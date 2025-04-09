from staticfg import CFGBuilder
from typing import Tuple, List, Dict
import copy
from dataclasses import dataclass
import numpy as np

from verse.agents.base_agent import BaseAgent
from verse.analysis import Simulator, Verifier, AnalysisTreeNode, AnalysisTree, ReachabilityMethod
from verse.analysis.analysis_tree import AnalysisTreeNodeType
from verse.utils.utils import sample_rect
from verse.parser.parser import ControllerIR
from verse.sensor.base_sensor import BaseSensor
from verse.map.lane_map import LaneMap

cfg = CFGBuilder().build_from_file('mylib', '/Users/bachhoang/Verse-library/verse/scenario/scenario.py')
cfg.build_visual('mylib', 'png')