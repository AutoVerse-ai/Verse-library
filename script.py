# def transpose(matrix):
#     return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

# matrix = [[0.5, 0.5, 0, 0],[0.55, 0.55, 0, 0]]
# #sum = -1.3e-06 + -1.3e-06

# # transposed_matrix = transpose(matrix)
# # print(transposed_matrix)
# #print(sum)

# # init_h = [[10 for _ in range(28)],[11 for _ in range(28)]]

# # print(init_h)

# matrix = [[row[0],row[3]] for row in matrix]
# print(matrix)

import matplotlib.pyplot as plt
import numpy as np

def sup(name):
    # Example data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    plt.plot(x, y)

    # Save the plot to a file
    plt.savefig(name)  # You can change the file name and format (e.g., 'my_plot.pdf')

    # Close the plot
    plt.close()

sup('hi.png')