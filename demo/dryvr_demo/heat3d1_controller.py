from enum import Enum, auto
import copy

class AgentMode(Enum):
    Default = auto()

class State:
    x_0 = 0.0
    x_1 = 0.0
    x_2 = 0.0
    x_3 = 0.0
    x_4 = 0.0
    x_5 = 0.0
    x_6 = 0.0
    x_7 = 0.0
    x_8 = 0.0
    x_9 = 0.0
    x_10 = 0.0
    x_11 = 0.0
    x_12 = 0.0
    x_13 = 0.0
    x_14 = 0.0
    x_15 = 0.0
    x_16 = 0.0
    x_17 = 0.0
    x_18 = 0.0
    x_19 = 0.0
    x_20 = 0.0
    x_21 = 0.0
    x_22 = 0.0
    x_23 = 0.0
    x_24 = 0.0
    x_25 = 0.0
    x_26 = 0.0
    x_27 = 0.0
    x_28 = 0.0
    x_29 = 0.0
    x_30 = 0.0
    x_31 = 0.0
    x_32 = 0.0
    x_33 = 0.0
    x_34 = 0.0
    x_35 = 0.0
    x_36 = 0.0
    x_37 = 0.0
    x_38 = 0.0
    x_39 = 0.0
    x_40 = 0.0
    x_41 = 0.0
    x_42 = 0.0
    x_43 = 0.0
    x_44 = 0.0
    x_45 = 0.0
    x_46 = 0.0
    x_47 = 0.0
    x_48 = 0.0
    x_49 = 0.0
    x_50 = 0.0
    x_51 = 0.0
    x_52 = 0.0
    x_53 = 0.0
    x_54 = 0.0
    x_55 = 0.0
    x_56 = 0.0
    x_57 = 0.0
    x_58 = 0.0
    x_59 = 0.0
    x_60 = 0.0
    x_61 = 0.0
    x_62 = 0.0
    x_63 = 0.0
    x_64 = 0.0
    x_65 = 0.0
    x_66 = 0.0
    x_67 = 0.0
    x_68 = 0.0
    x_69 = 0.0
    x_70 = 0.0
    x_71 = 0.0
    x_72 = 0.0
    x_73 = 0.0
    x_74 = 0.0
    x_75 = 0.0
    x_76 = 0.0
    x_77 = 0.0
    x_78 = 0.0
    x_79 = 0.0
    x_80 = 0.0
    x_81 = 0.0
    x_82 = 0.0
    x_83 = 0.0
    x_84 = 0.0
    x_85 = 0.0
    x_86 = 0.0
    x_87 = 0.0
    x_88 = 0.0
    x_89 = 0.0
    x_90 = 0.0
    x_91 = 0.0
    x_92 = 0.0
    x_93 = 0.0
    x_94 = 0.0
    x_95 = 0.0
    x_96 = 0.0
    x_97 = 0.0
    x_98 = 0.0
    x_99 = 0.0
    x_100 = 0.0
    x_101 = 0.0
    x_102 = 0.0
    x_103 = 0.0
    x_104 = 0.0
    x_105 = 0.0
    x_106 = 0.0
    x_107 = 0.0
    x_108 = 0.0
    x_109 = 0.0
    x_110 = 0.0
    x_111 = 0.0
    x_112 = 0.0
    x_113 = 0.0
    x_114 = 0.0
    x_115 = 0.0
    x_116 = 0.0
    x_117 = 0.0
    x_118 = 0.0
    x_119 = 0.0
    x_120 = 0.0
    x_121 = 0.0
    x_122 = 0.0
    x_123 = 0.0
    x_124 = 0.0

    agent_mode: AgentMode = AgentMode.Default

    def __init__(self, x, y,z ,agent_mode: AgentMode):
        pass

def decisionLogic(ego: State):
    output = copy.deepcopy(ego)

    return output