enum modes {NormalA, NormalB, NormalC, NormalD};

struct State {
int posx;
int posy;
int vx;
int vy;
enum modes mode;
};

	
	
struct State P(struct State s) {
	int posx = s.posx;
	int posy = s.posy;
	int vx = s.vx;
	int vy = s.vy;
	enum modes state = s.mode;
	if (state ==NormalA) {
		if (posy<0 && posy>=-0.01) {
			vy=-vy;
			posy=0;
			state=NormalA;
		}
		if (posx<0 && posx>=-0.01) {
			vx=-vx;
			posx=0;
			state=NormalB;
		}
		if (posx<=5.01 && posx>5) {
			vx=-vx;
			posx=5;
			state=NormalC;
		}
		if (posy>5 && posy<=5.01) {
			vy=-vy;
			posy=5;
			state = NormalD;
		}
	
	}
	if (state ==NormalB) {
		if (posy<0 && posy>=-0.01) {
			vy=-vy;
			posy=0;
			state=NormalB;
		}
		if (posx<0 && posx>=-0.01) {
			vx=-vx;
			posx=0;
			state=NormalA;
		}
		if (posx<=5.01 && posx>5) {
			vx=-vx;
			posx=5;
			state=NormalC;
		}
		if (posy>5 && posy<=5.01) {
			vy=-vy;
			posy=5;
			state = NormalD;
		}
	
	}
	if (state ==NormalC) {
		if (posy<0 && posy>=-0.01) {
			vy=-vy;
			posy=0;
			state=NormalC;
		}
		if (posx<0 && posx>=-0.01) {
			vx=-vx;
			posx=0;
			state=NormalA;
		}
		if (posx<=5.01 && posx>5) {
			vx=-vx;
			posx=5;
			state=NormalB;
		}
		if (posy>5 && posy<=5.01) {
			vy=-vy;
			posy=5;
			state = NormalD;
		}
	
	}
	
	if (state ==NormalD) {
		if (posy<0 && posy>=-0.01) {
			vy=-vy;
			posy=0;
			state=NormalD;
		}
		if (posx<0 && posx>=-0.01) {
			vx=-vx;
			posx=0;
			state=NormalA;
		}
		if (posx<=5.01 && posx>5) {
			vx=-vx;
			posx=5;
			state=NormalB;
		}
		if (posy>5 && posy<=5.01) {
			vy=-vy;
			posy=5;
			state = NormalC;
		}
	
	}
	s.posx = posx;
	s.posy= posy;
	s.vx = vx;
	s.vy = vy;
	s.mode = state;
	return s;

}
