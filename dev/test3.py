from scipy.optimize import minimize
from sympy import Symbol, diff
from sympy.utilities.lambdify import lambdify

# def func(arg):
#     x1, x2, x3, x4, w = arg 
#     return x1+0.01*(-2*x1+x2*(1+x1)+x3+w)

# def jac(arg):
#     x1, x2, x3, x4, w = arg 
#     dx1 = 1+0.01*(-2+x2)
#     dx2 = 0.01*(1+x1)
#     dx3 = 0.01
#     dx4 = 0
#     dw = 0.01
#     return [dx1, dx2, dx3, dx4, dw]

x1 = Symbol('x1',real=True)
x2 = Symbol('x2',real=True)
x3 = Symbol('x3',real=True)
x4 = Symbol('x4',real=True)
w = Symbol('w', real=True)

dt = 0.01

expr = x1 + 0.01*(-2*x1+x2*(1+x1)+x3+w)
vars = [x1, x2, x3, x4, w]
expr_func = lambdify([vars], expr)
jac = []
for var in vars:
    jac.append(diff(expr, var))
jac_func = lambdify([vars], jac)

x0 = [1.5, 1.5, 1, 0, 0]
# # expr1 = lambda x: x[0]+0.01*(-2*(x[0])+x[1]*(1+x[0])+x[2]+x[4])
res = minimize(
    expr_func, 
    x0, 
    bounds=((1,1.5),(1,1.5),(1,1),(0,0),(-0.1,0.1)), 
    jac = jac_func,
    # method='L-BFGS-B'
)
print(res.fun)