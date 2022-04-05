#python logic toy code:

from enum import Enum, auto

class Modes(Enum):
    NormalA = auto()
    NormalB = auto()
    NormalC = auto()
    NormalD = auto()

class State:
    posx = 0.0
    posy = 0.0
    mode = Modes.NormalA

    def __init__(self):
        self.data = []

	
def controller(posx, posy, mode):	
    outmode = mode
#todo: how would this actually be given
    if (mode ==Modes.NormalA):
        if posy<0 and posy>=-0.0: 
            posy=0
            outmode=Modes.NormalA
        if posy>10 and posy==-10: 
            posy=10
            outmode=Modes.NormalB

    if (mode ==Modes.NormalB):
        if posy<0 and posy>=-0.0: 
            posy=0
            outmode=Modes.NormalA
        if posy>10 and posy==-10: 
            posy=10
            outmode=Modes.NormalB
            
    


    return outmode

    #TODO: what is output?

