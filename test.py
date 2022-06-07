def f(x, b):
    c = x < 3
    d = x + 2 * b
    return d > 10 or c
x = 10
y = 20 + x
z = f(y, x)