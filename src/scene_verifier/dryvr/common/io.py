"""
This file contains IO functions for DryVR
"""

import six

from src.scene_verifier.dryvr.common.utils import DryVRInput, RrtInput, checkVerificationInput, checkSynthesisInput


def writeReachTubeFile(result, path):
    """
    Write reach tube to a file 
    
    reach tube example:
        mode1
        [0.0, 1, 2]
        [0.1, 2, 3]
        [0.1, 2, 3]
        ....
        mode1->mode2
        [1.0, 3, 4]
        ....
        
    Args:
        result (list): list of reachable state.
        path (str): file name.

    Returns:
        None

    """
    with open(path, 'w') as f:
        for line in result:
            if isinstance(line, six.string_types):
                f.write(line + '\n')
            elif isinstance(line, list):
                f.write(' '.join(map(str, line)) + '\n')


def writeRrtResultFile(modes, traces, path):
    """
    Write control synthesis result to a file 
    
    Args:
        modes (list): list of mode.
        traces (list): list of traces corresponding to modes
        path (str): file name.

    Returns:
        None

    """
    with open(path, 'w') as f:
        for mode, trace in zip(modes, traces):
            f.write(mode + '\n')
            for line in trace:
                f.write(" ".join(map(str, line)) + '\n')


def parseVerificationInputFile(data):
    """
    Parse the json input for DryVR verification
    
    Args:
        data (dict): dictionary contains parameters

    Returns:
        DryVR verification input object

    """

    # If resets is missing, fill with empty resets
    if not 'resets' in data:
        data['resets'] = ["" for _ in range(len(data["edge"]))]

    # If initialMode is missing, fill with empty initial mode
    if not 'initialVertex' in data:
        data['initialVertex'] = -1

    # If deterministic is missing, default to non-deterministic
    if not 'deterministic' in data:
        data['deterministic'] = False

    # If bloating method is missing, default global descrepancy
    if not 'bloatingMethod' in data:
        data['bloatingMethod'] = 'GLOBAL'

    # Set a fake kvalue since kvalue is not used in this case

    if data['bloatingMethod'] == "GLOBAL":
        data['kvalue'] = [1.0 for i in range(len(data['variables']))]

    # Set a fake directory if the directory is not provided, this means the example provides
    # simulation function to DryVR directly
    if not 'directory' in data:
        data['directory'] = ""

    checkVerificationInput(data)
    return DryVRInput(
        vertex=data["vertex"],
        edge=data["edge"],
        guards=data["guards"],
        variables=data["variables"],
        initialSet=data["initialSet"],
        unsafeSet=data["unsafeSet"],
        timeHorizon=data["timeHorizon"],
        path=data["directory"],
        resets=data["resets"],
        initialVertex=data["initialVertex"],
        deterministic=data["deterministic"],
        bloatingMethod=data['bloatingMethod'],
        kvalue=data['kvalue'],
    )


def parseRrtInputFile(data):
    """
    Parse the json input for DryVR controller synthesis
    
    Args:
        data (dict): dictionary contains parameters

    Returns:
        DryVR controller synthesis input object

    """

    # If bloating method is missing, default global descrepancy
    if not 'bloatingMethod' in data:
        data['bloatingMethod'] = 'GLOBAL'

    # set a fake kvalue since kvalue is not used in this case

    if data['bloatingMethod'] == "GLOBAL":
        data['kvalue'] = [1.0 for i in range(len(data['variables']))]

    # Set a fake directory if the directory is not provided, this means the example provides
    # simulation function to DryVR directly
    if not 'directory' in data:
        data['directory'] = ""

    checkSynthesisInput(data)

    return RrtInput(
        modes=data["modes"],
        variables=data["variables"],
        initialSet=data["initialSet"],
        unsafeSet=data["unsafeSet"],
        goalSet=data["goalSet"],
        timeHorizon=data["timeHorizon"],
        minTimeThres=data["minTimeThres"],
        path=data["directory"],
        goal=data["goal"],
        bloatingMethod=data['bloatingMethod'],
        kvalue=data['kvalue'],
    )
