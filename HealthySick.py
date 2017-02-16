"""
    Python code to analyse the Binary example.

    Alexander Erreygers, 2017
"""

#################################################
# Initial imports
#################################################
import numpy as np
import matplotlib.pyplot as plt
import ictmc

#################################################
# Declaring the used variables
#################################################

n = 2  # number of states
lam = np.array([[1/52, 3/52], [1/2, 2]])  # a_, a^ (h->s), b_, b^ (s->h)

t_final = 1  # Final time to end the simulation
num_plot = 40  # Number of points to plot, including 0

# Gamble to compute the lower prevision of
# Probability of being sick
gamble = np.array([0, 1])

#################################################
# Initialising the lower transition rate operator
#################################################

low_rate_op = ictmc.BinaryLowerTransitionRateOperator(lam)

# ##################################
# # Approximating the coefficient of ergodicity and the ergodic rate
# ##################################
# fig, ax = plt.subplots(2, sharex=True)
# low_rate_op.approximate_ergodic_rate(m=1, n=50, delta_max=None, ax=ax)
# plt.figure()
# ax = plt.gca()
# low_rate_op.long_term_uniform_error(m=1, n=70, delta_max=0.01, ax=ax)
# plt.show()

approximations = []
labels = []

# To determine the average computational time:
N = 50  # N = 0 does not do anything


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
    error = ictmc.compute_norm(approx.final_approx() - actual)
    print("The actual error is {}".format(error))
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


##################################
# Actual lower prevision
##################################
print_header("Determining the actual value of the lower previsions")
actual = low_rate_op.actual_conditional_lower_prevision(
    gamble, t_final)
print("The actual lower prevision is {}".format(actual))


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

##################################
# Limited number of iterations
##################################
maxit = 250
print_header("Uniform method with max {} iterations".format(maxit))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "UM",
    gamble, t_final, eps=eps, numplot=num_plot, method="uniform",
    tightererror=False, maxiter=maxit)

print_header("Uniform method with max {} iterations and e-keeping".format(
    maxit))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "UME",
    gamble, t_final, eps=eps, numplot=num_plot, method="uniform",
    tightererror=True, maxiter=maxit)

##################################
# Adaptive approximation
##################################
m = 1
print_header("Adaptive method with m = {}".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "A",
    gamble, t_final, eps=eps, numplot=num_plot, method="adaptive",
    tightererror=False, itersteps=m)

print_header("Adaptive method with m = {} and e-keeping".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "AE",
    gamble, t_final, eps=eps, numplot=num_plot, method="adaptive",
    tightererror=True, itersteps=m)

m = 20
print_header("Adaptive method with m = {}".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "A",
    gamble, t_final, eps=eps, numplot=num_plot, method="adaptive",
    tightererror=False, itersteps=m)

print_header("Adaptive method with m = {} and e-keeping".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "AE",
    gamble, t_final, eps=eps, numplot=num_plot, method="adaptive",
    tightererror=True, itersteps=m)

##################################
# Uniform and ergodic
##################################
m = 1
print_header("Uniform-ergodic method with m = {}".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "EU",
    gamble, t_final, eps=eps, numplot=num_plot, maxiter=None,
    method="uniform-ergodic", tightererror=False, itersteps=1)

print_header("Uniform-ergodic method with m = {} with e-keeping".format(m))
approximate_lowprev(
    low_rate_op.approx_conditional_lower_previsions, "EUE",
    gamble, t_final, eps=eps, numplot=num_plot, maxiter=None,
    method="uniform-ergodic", tightererror=True, itersteps=1)

##################################
# Limit value
##################################

print_header("Determining the limit")

actual_limit = low_rate_op.actual_limit_lower_prevision(gamble)
print("The actual limit is {}".format(actual_limit))

t_lim = 7
maxit = 250*7
eps = 1e-6
print("\n// Trying with the uniform method")
approx = low_rate_op.approx_conditional_lower_previsions(
            gamble, t_lim, eps=eps, numplot=num_plot, method="uniform",
            tightererror=False, maxiter=maxit)
approx.print_final_approx()
print("The actual error of the approximation is {}".format(
    ictmc.compute_norm([actual_limit, actual_limit] - approx.final_approx())))


# m = 1
# delta = 1e-7
# print("\n// Approximating the limit value with m = {}".format(m))
# approx_L = low_rate_op.approx_limit_lower_previsions(
#     gamble, eps=eps, itersteps=m)

# print("\n// Approximating the limit value with delta = {} and m = {}".format(
#     delta, m))
# approx_LD, eps_LD = low_rate_op.approx_limit_lower_previsions_alt(
#     gamble, delta, tightererror=True, itersteps=m)

#######
# Plots
#######

ictmc.compare_approximations_with_plots(approximations, labels, n, gamble)
