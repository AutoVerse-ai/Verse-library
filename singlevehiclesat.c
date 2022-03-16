enum modes {Normal,Sat_low,Sat_high};

struct State {
double tau;
double yf;
double thetaf;
enum modes mode;
};

	
	
struct State P(struct State s) {
	double tau = s.tau;
	double yf = s.yf;
	double thetaf = s.theta;
	enum modes state = s.mode;
	if (s.mode==Normal) {
		if (-0.155914*yf-thetaf <= -0.60871) {
			state=Sat_low;
		}
		if (-0.155914*yf-thetaf >= 0.60871) {
                        state=Sat_high;
                }
	
	}
	if (s.mode ==Sat_low) {
		if (-0.155914*yf-thetaf >= -0.60871) {
                        state=Normal;
                }
	
	}
	if (s.mode == Sat_high) {
		if (-0.155914*yf-thetaf <= 0.60871) {
                        state=Normal;
                }
	}
	
	}
	s.mode = state;
	return s;

}
