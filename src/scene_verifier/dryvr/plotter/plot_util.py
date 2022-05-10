import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def plot_rtsegment_and_traces(rtsegment: np.ndarray, traces: np.ndarray):
    for dim_ind in range(1, traces.shape[2]):
        fig, ax = plt.subplots(1)
        facecolor = 'r'
        for trace_ind in range(traces.shape[0]):
            ax.plot(traces[trace_ind, :, 0], traces[trace_ind, :, dim_ind])
        for hrect_ind in range(rtsegment.shape[0]):
            ax.add_patch(Rectangle((rtsegment[hrect_ind, 0, 0], rtsegment[hrect_ind, 0, dim_ind]), rtsegment[hrect_ind, 1, 0]-rtsegment[hrect_ind, 0, 0],
                                            rtsegment[hrect_ind, 1, dim_ind] - rtsegment[hrect_ind, 0, dim_ind], alpha=0.1, facecolor='r'))
        ax.set_title(f'dim #{dim_ind}')
        fig.canvas.draw()
        plt.show()


def plot_traces(traces, dim, bloat_tube):
    """ Plot the traces """
    # Iterate over all individual traces
    for i in range(0, len(traces)):
        trace = traces[i]

        # Obtain desired dimension
        time = []
        data = []
        for j in range(0, len(trace)):
            # for j in xrange(0,2):
            time.append(trace[j][0])
            data.append(trace[j][dim])

        # Plot data
        if i == 0:
            plt.plot(time, data, 'b')
        else:
            plt.plot(time, data, 'r')

    time = [row[0] for row in bloat_tube]
    value = [row[dim] for row in bloat_tube]
    time_bloat = [time[i] for i in range(0, len(value), 2)]
    lower_bound = [value[i] for i in range(0, len(value), 2)]
    upper_bound = [value[i + 1] for i in range(0, len(value), 2)]
    plt.plot(time_bloat, lower_bound, 'g')
    plt.plot(time_bloat, upper_bound, 'g')