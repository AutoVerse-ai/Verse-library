enum modes {go_forward, move_over};

struct State {
int vx;
int vy;
int dist;
//enum modes mode;
};

	
	
struct State P(struct State s) {
	int posx = s.posx;
	int posy = s.posy;
	int vx = s.vx;
	int vy = s.vy;
	int dist = s.dist
	enum modes state = s.mode;
	if (dist < 10) {
		if (dist < 2) {
			if (state != move_over) {
			//if super close, slow down and begin move over if we aren't already
                        	vy=vy -5;
                		vx= vx + 5;
			}
		}
		else if (dist > 2) {
			vx=vx - 5;
		}
	
	}
	else if (state == move_over) {
		vx = vx + 5;
	
	}
	s.vx = vx;
	s.vy = vy;
	s.mode = state;
	return s;

}
