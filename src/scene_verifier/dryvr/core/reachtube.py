"""
This file contains reach tube class for DryVR
"""

import six


class ReachTube:
    """
    This is class is an object for reach tube
    Ideally it should support to fetch reachtube by mode and variable name
    And it should allow users to plot the reach tube in different ways
    """

    def __init__(self, tube, variables, modes):
        """
            ReachTube class initialization function.

            Args:
            tube (list): raw reach tube (that used to print to file)
            variables (list): list of variables in the reach tube
            modes (list): list of modes in the reach ReachTube
        """
        self._tube = tube
        self._variables = variables
        self._modes = modes

        # Build the raw string representation so example can print it

        self.raw = ""
        for line in tube:
            if isinstance(line, str):
                self.raw += line + "\n"
            else:
                self.raw += " ".join(map(str, line)) + '\n'

        # Build dictionary object so you can easily pull out part of the list
        self._tube_dict = {}
        for mode in modes:
            self._tube_dict[mode] = {}
            for var in variables + ["t"]:
                self._tube_dict[mode][var] = []

        cur_mode = ""
        for line in tube:
            if isinstance(line, six.string_types):
                cur_mode = line.split('->')[-1].split(',')[0]  # Get current mode name
                for var in ['t'] + self._variables:
                    self._tube_dict[cur_mode][var].append(line)
            else:
                for var, val in zip(['t'] + self._variables, line):
                    self._tube_dict[cur_mode][var].append(val)

    def __str__(self):
        """
            print the raw tube
        """
        return str(self.raw)

    def filter(self, mode=None, variable=None, contain_label=True):
        """
            This is a filter function that allows you to select 
            Args:
            mode (str, list): single mode name or list of mode name
            variable (str, list): single variable or list of variables
        """
        if mode is None:
            mode = self._modes
        if variable is None:
            variable = ["t"] + self._variables

        if isinstance(mode, str):
            mode = [mode]
        if isinstance(variable, str):
            variable = [variable]

        res = []
        for m in mode:
            temp = []
            for i in range(len(self._tube_dict[m]["t"])):
                if isinstance(self._tube_dict[m]["t"][i], str):
                    if contain_label:
                        temp.append([self._tube_dict[m]["t"][i]] + variable)
                    continue

                temp.append([self._tube_dict[m][v][i] for v in variable])
            res.append(temp)
        return res
