"""
    An implementation of the ICTMC discussed in (Troffaes et al, 2015).

    Alexander Erreygers, 2016
"""

#################################################
# Initial imports
#################################################
import numpy as np
# from scipy.optimize import linprog
import cvxopt
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import math
import ictmc
# import functools

plt.rcParams['image.cmap'] = 'viridis'

#################################################
# Declaring the used variables
#################################################


Qlow = np.array([[-0.98, 0.32, 0.32, 0.19],
                 [730, -1460.61, 0, 0.51],
                 [730, 0, -1460.61, 0.51],
                 [0, 730, 730, -2920]])
Qup = np.array([[-0.83, 0.37, 0.37, 0.24],
                [1460, -730.51, 0, 0.61],
                [1460, 0, -730.51, 0.61],
                [0, 1460, 1460, -1460]])
n = 4

t_final = 0.02    # Final time to end the simulation
num_plot = 30    # Number of points to plot, including 0

# Gamble to compute the lower prevision of
#   Probability of ending up in some state
gamble = np.zeros(n)
gamble[0] = 1

#################################################
# Initialising the lower transition rate operator
#################################################

# Improving the linear program solver
# https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html

# These matrices allow for more efficient computation
A = cvxopt.matrix([1. for i in range(n)], (1, n))
b = cvxopt.matrix([0.], (1, 1))
G_array = []
h_array = []
for i in range(n):
    vals, x, y, h = [], [], [], []
    ind = 0
    for j in range(n):
        vals.append(1)
        x.append(ind)
        y.append(j)
        h.append(Qup[i, j])
        ind += 1
        vals.append(-1)
        x.append(ind)
        y.append(j)
        h.append(- Qlow[i, j])
        ind += 1
    G_array.append(cvxopt.spmatrix(vals, x, y, (2*n, n)))
    h_array.append(cvxopt.matrix(h))


# The actual solution method
cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8

_linprog = cvxopt.solvers.lp


def apply_lower_rate(g):
    try:
        ret = np.empty(n)
        c = cvxopt.matrix(g.astype(float).tolist(), (n, 1))
        for i in range(n):
            sol = _linprog(c, G_array[i], h_array[i], A, b, solver='glpk')
            ret[i] = np.array(sol['x']).reshape(n).dot(g)
        return ret
    except:
        print("error")
        return g

# If you do not have GLPK and/or cvxopt, you can use numpy
# However, this will be terribly slow!
# def apply_lower_rate_old(g):
#     try:
#         g = np.copy(g)
#         ret = np.empty(n)
#         for i in range(n):
#             Qbounds = [(Qlow[i, j], Qup[i, j]) for j in range(0, n)]
#             lpRes = linprog(g, A_eq=np.ones((1, n)), b_eq=[0],
#                             bounds=Qbounds)
#             ret[i] = lpRes.fun
#         return ret
#     except:
#         print("error")
#         return g


# low_rate_op_old = ictmc.LowerTransitionRateOperator(apply_lower_rate_old, n)
low_rate_op = ictmc.LowerTransitionRateOperator(apply_lower_rate, n)


approximations = []
labels = []

# To determine the average computational time:
N = 0  # N = 0 does not do anything


def determine_average(func, *args, **kwargs):
    print('{:*^60}'.format(''))
    print('{:*^60}'.format(" Determining average duration. "))
    print('{:*^60}'.format(''))
    durations = []
    for i in range(N):
        durations.append(func(*args, **kwargs).total_duration())
    average_duration = sum(durations) / N
    print('{:*^60}'.format(''))
    print('{:*^60}'.format(" Average duration of computations: {} ".format(
        average_duration)))
    print('{:*^60}'.format(''))


def approximate_lowprev(func, label, *args, **kwargs):
    approx = func(*args, **kwargs)
    approx.print_final_approx()
    approximations.append(approx)
    labels.append(label)
    # Getting an average
    if N > 0:
        determine_average(func, *args, **kwargs)


def print_header(title):
    print("\n")
    print('{:X^80}'.format(''))
    print('{:x^80}'.format(' ' + title + ' '))
    print('{:X^80}'.format(''))


# ##################################
# # Approximating the coefficient of ergodicity and the ergodic rate
# ##################################
# # fig, ax = plt.subplots(2, sharex=True)
# # delta, coefferg, ergrate = low_rate_op.approximate_ergodic_rate(
# #     m=4, n=35, delta_max=1e-8, ax=ax)
# # # Hier is m=1 ook niet voldoende!
# # plt.legend()
# # plt.show()

eps = 1e-3
##################################
# Original numerical approximation with limited number of iterations
##################################
maxit = 80
print_header("Uniform method with max {} iterations".format(maxit))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "UM",
    gamble, t_final, eps=eps, numplot=num_plot, method="uniform",
    tightererror=False, maxiter=maxit)

eps = 1e-3
##################################
# Uniform numerical approximation
##################################
print_header("Uniform method")
approximate_lowprev(
        low_rate_op.approx_conditional_lower_previsions, "U",
        gamble, t_final, eps=eps, numplot=num_plot, method="uniform",
        tightererror=False)

print_header("Uniform method with e-keeping")
approximate_lowprev(
        low_rate_op.approx_conditional_lower_previsions, "UE",
        gamble, t_final, eps=eps, numplot=num_plot, method="uniform",
        tightererror=True)

eps = 1e-3
##################################
# Adaptive approximation
##################################
m = 3
print_header("Adaptive method with m = {} and e-keeping".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "A",
    gamble, t_final, eps=eps, numplot=num_plot, method="adaptive",
    tightererror=False, itersteps=m)

m = 24
print_header("Adaptive method with m = {} and e-keeping".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "A",
    gamble, t_final, eps=eps, numplot=num_plot, method="adaptive",
    tightererror=False, itersteps=m)

eps = 1e-3
##################################
# Uniform and ergodic
##################################
m = 3
print_header("Uniform-ergodic method with m = {}".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "EU",
    gamble, t_final, eps=eps, numplot=num_plot, maxiter=None,
    method="uniform-ergodic", tightererror=False, itersteps=1)

##################################
# Limit value
##################################
eps = 1e-6
m = 3
delta = 1e-8

print_header("Approximating the limit value with m = {}".format(m))
approx_L = low_rate_op.approx_limit_lower_previsions(
    gamble, eps=eps, itersteps=m)

print_header("Approximating the limit value with delta = {}".format(delta))
approx_LD, eps_LD = low_rate_op.approx_limit_lower_previsions_alt(
    gamble, delta, tightererror=True, itersteps=m)

#######
# Plots
#######

ictmc.compare_approximations_with_plots(approximations, labels, n, gamble)
