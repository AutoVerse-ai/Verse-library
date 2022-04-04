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
    outstate = mode
#todo: how would this actually be given
    if (state ==Modes.NormalA):
        if posy<0 and posy>=-0.01: 
            posy=0
            outstate=Modes.NormalA
        if posy>0 and posy==-0.01: 
            posy=0
            outstate=Modes.NormalB
            
    


    return outstate

    #TODO: what is output?

