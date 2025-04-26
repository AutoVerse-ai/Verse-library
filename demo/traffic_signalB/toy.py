import pyvista as pv

plotter = pv.Plotter()
plotter.add_mesh(pv.Box(bounds=[-2000, 200, -10, 100, 0,0]), color='red')

plotter.show_grid(n_xlabels=100)
plotter.show()
print(pv.Report())