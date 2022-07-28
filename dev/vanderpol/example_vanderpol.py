import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt 

def dynamics1(t, state, u):
    x1, x2 = state 
    w1, w2 = u
    x1_dot = w1*x2 
    x2_dot = (1-x1**2)*x2*w2-x1
    return [x1_dot, x2_dot]

def dynamics2(t, state, u):
    x1, x2, x1_hat, x2_hat = state 
    x1_dot = 1.1*x2 - 2000.0
    x2_dot = 2.2e8*x1 - 2.2e8*x1_hat - 1.1*x2*(x1**2 - 1.0) - 2000.0
    
    x1_hat_dot = 1.1*x2_hat - 0.2*x2 + 2000.0
    x2_hat_dot =  2.2e8*x1_hat - 2.2e8*x1 - 0.9*x2_hat*(x1_hat**2 - 1.0) + 2000.0
    return [x1_dot, x2_dot, x1_hat_dot, x2_hat_dot]

def TC_simulate1(mode, initialCondition, time_bound, time_step, noise=False):
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound/time_step))
    t = [i*time_step for i in range(0,number_points)]

    init = initialCondition 
    trace = [[0] + init]
    for i in range(len(t)):
        if noise:
            w1 = np.random.uniform(0.9, 1.1)
            w2 = np.random.uniform(0.9, 1.1)
        else:
            w1 = 1
            w2 = 1
        r = ode(dynamics1)
        r.set_initial_value(init).set_f_params([w1,w2])
        res:np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init) 
    return trace 

def TC_simulate2(mode, initialCondition, time_bound, time_step):
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound/time_step))
    t = [i*time_step for i in range(0,number_points)]

    init = initialCondition 
    trace = [[0] + init]
    for i in range(len(t)):
        w1, w1_hat = 0.9, 1.1
        w2, w2_hat = 0.9, 1.1
        r = ode(dynamics2)
        r.set_initial_value(init).set_f_params([w1, w2, w1_hat, w2_hat])
        res:np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init) 
    return trace 

if __name__ == "__main__":
    for i in range(10):
        res = TC_simulate1(
            None, 
            [1.4, 2.3],
            10,
            0.1,
            True 
        )
        res = np.array(res)
        plt.figure(0)
        plt.plot(res[:,0], res[:,1], 'b')

        plt.figure(1)
        plt.plot(res[:,0], res[:,2], 'b')

        plt.figure(2)
        plt.plot(res[:,1], res[:,2], 'b')

    # res = TC_simulate2(
    #     None, 
    #     [1.4, 2.3, 1.4, 2.3],
    #     100,
    #     0.01,
    # )

    # res = np.array(res)

    # plt.figure(0)
    # plt.plot(res[:,0], res[:,1],'r')
    # plt.plot(res[:,0], res[:,3],'g')
    # plt.title('x1')

    # plt.figure(1)
    # plt.plot(res[:,0], res[:,2],'r')
    # plt.plot(res[:,0], res[:,4],'g')
    # plt.title('x2')
    
    plt.show()