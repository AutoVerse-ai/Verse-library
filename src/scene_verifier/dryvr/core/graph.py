"""
This file contains graph class for DryVR
"""
# import matplotlib.pyplot as plt
import networkx as nx


class Graph:
    """
    This is class to plot the progress graph for dryvr's
    function, it is supposed to display the graph in jupyter
    notebook and update the graph as dryvr is running
    """

    def __init__(self, params, is_ipynb):
        """
        Guard checker class initialization function.

        Args:
            params (obj): An object contains the parameter
            is_ipynb (bool): check if the code is running on ipython or not
        """
        self._is_ipynb = is_ipynb
        if not is_ipynb:
            return
        vertex = []
        # Build unique identifier for a vertex and mode name
        for idx, v in enumerate(params.vertex):
            vertex.append(v + "," + str(idx))

        edges = params.edge
        self.edgeList = []
        for e in edges:
            self.edgeList.append((vertex[e[0]], vertex[e[1]]))
        # Initialize the plot
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)

        # Initialize the graph
        self.G = nx.DiGraph()
        self.G.add_edges_from(self.edgeList)
        self.pos = nx.spring_layout(self.G)
        self.colors = ['green'] * len(self.G.nodes())
        # self.fig.suptitle('', fontsize=10)
        # Draw the graph when initialize
        # self.draw()
        # plt.show()

    def draw(self):
        """
        Draw the white-box transition graph.
        """

        nx.draw_networkx_labels(self.G, self.pos)
        options = {
            'node_color': self.colors,
            'node_size': 1000,
            'cmap': plt.get_cmap('jet'),
            'arrowstyle': '-|>',
            'arrowsize': 50,
        }
        nx.draw_networkx(self.G, self.pos, arrows=True, **options)
        self.fig.canvas.draw()

    def update(self, cur_node, title, remain_time):
        """
        update the graph 
        Args:
            cur_node (str): current vertex dryvr is verifying
            title (str): current transition path as the title
            remain_time (float): remaining time
        """
        if not self._is_ipynb:
            return
        self.ax.clear()
        self.colors = ['green'] * len(self.G.nodes())
        self.colors[list(self.G.nodes()).index(cur_node)] = 'red'
        self.fig.suptitle(title, fontsize=10)
        self.ax.set_title("remain:" + str(remain_time))
        self.draw()
