"""
This file consist parser code for DryVR reachtube output
"""

from typing import TextIO
import re 

class Parser:
    def __init__(self, f: TextIO):
        data = f.readlines()
        curr_key = ""
        self.data_dict = {}
        i = 0
        while i < len(data):
            line = data[i]
            if not re.match('^[-+0-9+.+0-9+e+0-9 ]+$', line):
                self.data_dict[line] = []
                curr_key = line
                i += 1
            else:
                line_lower = data[i]
                line_lower_list = line_lower.split(' ')
                line_lower_list = [float(elem) for elem in line_lower_list]
                line_upper = data[i+1]
                line_upper_list = line_upper.split(' ')
                line_upper_list = [float(elem) for elem in line_upper_list]
                rect = [line_lower_list, line_upper_list]
                self.data_dict[curr_key].append(rect)
                i += 2
                
    
    def get_all_data(self):
        res = []
        for key in self.data_dict:
            res += self.data_dict[key]
        return res

    