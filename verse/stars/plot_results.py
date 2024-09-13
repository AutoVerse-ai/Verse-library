import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = pd.read_csv('./verse/stars/nn_results.csv').to_numpy()
plt.plot(results[:, 0], results[:, 1]/max(results[:,1]), label='mu')
plt.plot(results[:, 0], results[:, 2]/max(results[:,2]), label='Percent Containment')
plt.xlabel('Time (s)')
plt.legend()
plt.show()