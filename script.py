def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

matrix = [[0.5, 0.5, 0, 0],[0.55, 0.55, 0, 0]]
#sum = -1.3e-06 + -1.3e-06

transposed_matrix = transpose(matrix)
print(transposed_matrix)
#print(sum)

# init_h = [[10 for _ in range(28)],[11 for _ in range(28)]]

# print(init_h)