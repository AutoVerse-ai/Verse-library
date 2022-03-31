#python logic toy code:

from enum import Enum, auto
from ssl import VERIFY_X509_PARTIAL_CHAIN
from statistics import mode
from termios import VSTART


class Modes(Enum):
    NormalA = auto()
    NormalB = auto()
    NormalC = auto()
    NormalD = auto()

class State:
    posx = 0.0
    posy = 0.0
    vx = 1.0
    vy = 1.0
    mode = Modes.NormalA

    def __init__(self):
        self.data = []

	
	
#todo: how would this actually be given
s = State()

posx = s.posx
posy = s.posy
vx = s.vx
vy = s.vy
state = s.mode
if (state ==Modes.NormalA):
	if posy<0 and posy>=-0.01: 
		vy=-vy
		posy=0
		state=Modes.NormalA
		
	if (posx<0 and posx>=-0.01) :
			vx=-vx
			posx=0
			state=Modes.NormalB
		
	if (posx<=5.01 and posx>5) :
		vx=-vx
		posx=5
		state=Modes.NormalC
		
	if (posy>5 and posy<=5.01) :
		vy=-vy
		posy=5
		state = Modes.NormalD
		
	
	
if (state ==Modes.NormalB) :
	if (posy<0 and posy>=-0.01) :
		vy=-vy
		posy=0
		state=Modes.NormalB
		
	if (posx<0 and posx>=-0.01) :
    	vx=-vx
		posx=0
		state=Modes.NormalA
		
	if (posx<=5.01 and posx>5) :
		vx=-vx
		posx=5
		state=Modes.NormalC
		
	if (posy>5 and posy<=5.01) :
		vy=-vy
		posy=5
		state = Modes.NormalD
		
	
	
if (state ==Modes.NormalC) :
	if (posy<0 and posy>=-0.01) :
		vy=-vy
		posy=0
		state=Modes.NormalC
		
	if (posx<0 and posx>=-0.01) :
		vx=-vx
		posx=0
		state=Modes.NormalA
		
	if (posx<=5.01 and posx>5) :
		vx=-vx
		posx=5
		state=Modes.NormalB
		
	if (posy>5 and posy<=5.01) :
		vy=-vy
		posy=5
		state = Modes.NormalD
		
	
	
	
if (state ==State.Modes.NormalD) :
	if (posy<0 and posy>=-0.01) :
		vy=-vy
		posy=0
		state=Modes.NormalD
		
	if (posx<0 and posx>=-0.01) :
		vx=-vx
		posx=0
		state=Modes.NormalA
		
	if (posx<=5.01 and posx>5) :
		vx=-vx
		posx=5
		state=Modes.NormalB
		
	if (posy>5 and posy<=5.01) :
		vy=-vy
		posy=5
		state = Modes.NormalC
		
	
	
s.posx = posx
s.posy= posy
s.vx = vx
s.vy = vy
s.mode = state

#TODO: what is output?

