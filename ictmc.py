# Copyright (C) 2017  Alexander Erreygers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
    The ictmc Python modulde.

    This Python module can be used to approximate coherent lower (and upper)
    lower previsions for imprecise continuous-time Markov chains. This module
    contains naive, non-optimised Python implementations of the algorithms
    described in [1] and [2].

    References
    ----------
    .. [1] Alexander Erreygers and Jasper De Bock. ``Imprecise Continuous-Time
           Markov Chains: Efficient Computational Methods with Guaranteed Error
           Bounds''. arXiv:1702.07150.
    .. [2] Jasper De Bock. ``The Limit Behaviour of Imprecise Continuous-Time
           Markov Chains''.  Journal of Nonlinear Science (2017) 27:159.
           doi:10.1007/s00332-016-9328-3
"""

import numpy as np
import math
import time
import datetime
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product


def compare_approximations_with_plots(approxlist, labellist, dim, gamble,
                                      show=True):
    """Plot a number of different approximations of the same conditional lower
    prevision.

    Parameters
    -----
    approxlist : list of approx
        A list of approximations.
    labellist : list of str
        A list of labels that has the same lenght as `approxlist`.
    dim : int
        The dimension of the state space
    gamble : array_like
        The function of which the expectation is approximated.

    show : bool, optional
        If `True`, then the plt.show() command is executed.
    """
    numapprox = len(approxlist)
    if len(approxlist) is not len(labellist):
        print("ERROR.")
    c = cm.viridis(np.linspace(0, 1, num=dim))
    fig, ax = plt.subplots(numapprox, sharey=True, sharex=True)
    fig.suptitle("Conditional lower prevision of f = {}".format(gamble))
    for i in range(0, numapprox):
        approxlist[i].plot_lowerprevision(ax=ax[i], prefix=labellist[i])
        ax[i].legend()
        ax[i].set_xlabel("Time t")

    # Comparison of number of iterataions used
    c = cm.viridis(np.linspace(0, 1, num=numapprox))
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle("Comparison of the number of iterations")
    for i in range(0, numapprox):
        approxlist[i].plot_numiter(ax=ax[0], c=c[i], label=labellist[i])
        approxlist[i].plot_timestep(ax=ax[1], c=c[i], label=labellist[i])
    ax[0].set_xlabel("Time t")
    ax[0].set_ylabel("Cumulative number of iterations")
    ax[0].legend()
    ax[1].set_xlabel("Time t")
    ax[1].set_ylabel("Timestep delta")
    ax[1].legend()
    if show:
        plt.show()


def compute_norm(f):
    """Compute the maximum norm of the function f.

    Parameters
    ---------
    f : array_like
        A function to compute the maximum norm of.

    Returns
    -------
    float
        The maximum norm of `f`.
    """
    norm = max(np.abs(f))
    return norm


def compute_centred_norm(f):
    """Compute the centred norm of a function.

    Parameters
    ---------
    f : array_like
        A function to compute the centred norm of.

    Returns
    -------
    float
        The centred norm of `f`.
    """
    norm = (max(f) - min(f))/2
    return norm


class LowerTransitionRateOperator:
    """A lower transition rate operator.

    The `LowerTransitionRateOperator` class contains methods to check for
    ergodicity, approximate conditional lower previsions, and determine
    approximations of the coefficient of ergodicity of the approximation.
    """
    def __init__(self, lower_rate_fun, dim):
        """Initialise a `LowerTransitionRateOperator`.

        Parameters
        ----------
        lower_rate_fun : method
            A method that takes a function f (in the form of an `array_like`
             object), and returns Q f.
        dim : int
            The size of the state space.
        """
        print("*** Initialising the lower transition rate operator. ***")
        self.dim = dim
        self.eval = lower_rate_fun
        # Check for ergodicity
        self.is_ergodic = self.check_ergodicity()
        print("The lower transition rate operator is {wn}ergodic".format(
            wn="" if self.is_ergodic else "not "))
        self.norm = self.compute_lower_rate_norm()
        print("The norm of the lower transition rate operator is {}".format(
            self.norm))
        # Largest step size for which (I + delta Q) is still a lower transition
        # operator
        if self.norm > 0:
            self.deltamax = 2 / self.norm
        else:
            math.inf

    def compute_lower_rate_norm(self):
        """ Compute the norm of the lower transition rate operator.

        Based on Proposition 4 of [1].

        Returns
        -------
        float
            The norm of the lower transition rate operator.
        """
        norm = []
        eye = np.identity(self.dim)
        for i in range(0, self.dim):
            ind = eye[i, :]
            norm.append(2 * abs(self.eval(ind)[i]))
        norm = max(norm)
        return norm

    def check_upper_reachability(self):
        """Check if Q is top class regular.

        This method implements Algorithm 1 of [2].

        Returns
        -------
        ndarray
            A `dim` by `dim` matrix, where the (x,y) component is 1 if
            y is upper reachable from x, and 0 otherwise.

        """
        UR = np.zeros((self.dim, self.dim))
        eye = np.identity(self.dim)

        for x in range(0, self.dim):
            for y in range(0, self.dim):
                if x is y:
                    UR[x, y] = 1
                elif self.eval(-eye[x, :])[y] < 0:
                    UR[x, y] = 1

        for x in range(0, self.dim):
            for y in range(0, self.dim):
                for z in range(0, self.dim):
                    if UR[x, z] is 1 and UR[z, y] is 1:
                        UR[x, y] = 1
        return UR

    def check_lower_reachability(self, A):
        """Check if A is lower reachable.

        This method implements Algorithm 2 of [2].

        Parameters
        ----------
        A : list
            A list of indices corresponding to the states in the event A.

        Returns
        -------
        list
            A list of indices, corresponding to all the states x such that
            A is lower reachable from x.

        """
        B = A
        eye = np.identity(self.dim)

        S = [0]
        while len(S) > 0:
            S = []
            indB = np.zeros(self.dim)
            for x in B:
                indB = np.add(indB, eye[x, :])
            for y in range(0, self.dim):
                if y not in B:
                    if self.eval(indB)[y] > 0:
                        S.append(y)
            B.append(S)
        return B

    def check_ergodicity(self):
        """Check if the lower transition rate operator is ergodic.

        This method implements Algorithm 3 of [2].

        Returns
        -------
        bool
            `True` if the lower transition rate operator is ergodic, `False`
            otherwise.
        """
        print("// Checking ergodicity.")
        UR = self.check_upper_reachability()
        X1A = []
        for x in range(self.dim):
            test = True
            for y in range(self.dim):
                if UR[x, y] is 0:
                    test = False
                    break
            if test:
                X1A.append(x)
            else:
                return False
        print("X_1A = {}".format(X1A))
        if len(X1A) > 0:
            B = self.check_lower_reachability(X1A)
            test = True
            for i in range(0, self.dim):
                if i not in B:
                    test = False
                    break
            if test:
                return True
            else:
                return False

    def _determine_coeff_of_ergod(self, delta, rep=1):
        """Determine a lower and upper bound for the coefficient of ergodicity.

        This method implements Theorem 10 of [1], and determines a lower and
        upper bound for the coefficient of ergodicity of (I + delta Q)^{m}.

        Parameters
        ----------
        delta : float
            The length of the step size.
        rep : int, optional
            The value of m, the default is 1.

        Returns
        -------
        coeff : list of float
            `coeff[0]` is a guaranteed lower bound for the coefficient of
            ergodicity of (I + `delta` Q)^{`rep`}, and `coeff[1]` is a
            guaranteed upper bound.
        """
        coeff = [0, 0]
        poss = [0, 0]
        # Iterate over all non-trival events
        for indicatorA in product([0, 1], repeat=self.dim):
            retlow = np.copy(indicatorA)
            retup = np.copy(indicatorA)
            for i in range(rep):
                retlow = np.add(retlow, delta*self.eval(retlow))
                retup = np.add(retup, -delta*self.eval(-retup))
            # Possible new lower bound
            poss[0] = np.max(retlow) - np.min(retlow)
            # Possible new upper bound
            poss[1] = np.max(retup) - np.min(retlow)
            if coeff[0] < poss[0]:
                coeff[0] = poss[0]
            if coeff[1] < poss[1]:
                coeff[1] = poss[1]
        return coeff

    def approximate_coeff_of_ergod(self, m=0, n=50, delta_max=None, ax=None):
        """Approximate the coefficient of ergodicity for several delta's.

        We approximate the coefficient of ergodicity for several
        uniformly-spaced values of delta, and plot these approximations.

        Parameters
        ----------
        m : int, optional
            The number of repeated applications of Phi. If `m = 0`, then
            the dimension of the state space is used.
        n : int, optional
            The number of deltas in between 0 and `delta_max`
        delta_max : float, optional
            The maximal value of delta. If `delta_max` is equal to `None`,
            then the maximally allowable value (`self.deltamax`) is used.
        ax : axis, optional
            A `pyplot` plot axis, if we want to plot the results

        Returns
        -------
        delta : ndarray
            A `ndarray` of shape `(1,n+2)` that contains all the delta's
            that were used for the approximation.
        coeffs_of_ergod : ndarray
            A `ndarray` of shape `(2, n+2)`; the first row contains the
            lower bounds, the second the upper bounds.

        """
        print("// Determining bounds for the coefficient of ergodicity.")
        if m is 0:
            _m = self.dim - 1
        else:
            _m = m
        if self.norm is 0:
            return math.nan, 1
        elif delta_max is None:
            deltamax = self.deltamax
        else:
            deltamax = min(delta_max, self.deltamax)
        coeffs_of_ergod = np.empty((2, n+2))
        delta = np.linspace(0, deltamax, num=n+2)
        coeffs_of_ergod[:, 0] = [1, 1]  # delta = 0 => rho(I) = 1

        for i in tqdm(range(1, n+2)):
            coeffs_of_ergod[:, i] = self._determine_coeff_of_ergod(
                delta[i], rep=_m)
        # Check whether we need to plot the coefficients of ergodicity
        if ax is not None:
            ax.plot(delta, coeffs_of_ergod[0], '.-', label="Lower bound")
            ax.plot(delta, coeffs_of_ergod[1], '.-', label="Upper bound")
            ax.set_xlabel("delta")
            ax.set_ylabel("Bound for rho(I + {} Q){n}".format(
                delta, n="^{}".format(_m) if _m > 0 else ""))
        return delta, coeffs_of_ergod

    def approximate_ergodic_rate(self, m=0, n=50, delta_max=None, ax=None):
        """Approximate the ergodic rate using several delta's.

        We approximate the coefficient of ergodicity for several
        uniformly-spaced values of delta, and use these to approximate
        thee ergodic rate. If desired, then we also plot these approximations.

        Parameters
        ----------
        m : int, optional
            The number of repeated applications of Phi. If `m = 0`, then
            `dim - 1` is used.
        n : int, optional
            The number of deltas in between 0 and `delta_max`
        delta_max : float, optional
            The maximal value of delta. If `delta_max` is equal to `None`,
            then the maximally allowable value (`self.deltamax`) is used.
        ax : axis, optional
            A `pyplot` plot axis, if we want to plot the results

        Returns
        -------
        delta : ndarray
            A `ndarray` of shape `(1,n+2)` that contains all the delta's
            that were used for the approximation.
        coeffs_of_ergod : ndarray
            A `ndarray` of shape `(2, n+2)`; the first row contains the
            lower bounds for the coefficient of ergodicity, the second the
            upper bounds.
        coeffs_of_ergod : ndarray
            A `ndarray` of shape `(2, n+2)`; the first row contains the
            upper bounds for the ergodic rate, the second the lower bounds.
        """
        print("// Determining bounds for the ergodic rate.")
        if m is 0:
            _m = self.dim - 1
        else:
            _m = m
        delta, coeffs_of_ergod = self.approximate_coeff_of_ergod(
            m=_m, n=n, delta_max=delta_max, ax=None)
        erg_rate = np.divide(1 - coeffs_of_ergod[:, 1:], _m * delta[1:])
        # Check whether we need to plot the coefficients of ergodicity
        if ax is not None:
            if len(ax) is 1:
                ax.plot(delta, coeffs_of_ergod[0], '.-', label="lower")
                ax.plot(delta, coeffs_of_ergod[1], '.-', label="upper")
                ax.set_xlabel("delta")
                ax.set_ylabel("Bound for rho(I + delta Q){n}".format(
                    n="^{}".format(_m) if _m > 0 else ""))
            elif len(ax) is 2:
                ax[0].plot(delta, coeffs_of_ergod[0], '.-', label="lower")
                ax[0].plot(delta, coeffs_of_ergod[1], '.-', label="upper")
                ax[0].set_xlabel("delta")
                ax[0].set_ylabel("Bound for rho(I + delta Q){n}".format(
                    n="^{}".format(_m) if _m > 0 else ""))
                ax[1].plot(delta[1:], erg_rate[0], '.-', label="upper")
                ax[1].plot(delta[1:], erg_rate[1], '.-', label="lower")
                ax[1].set_xlabel("delta")
                ax[1].set_ylabel("Bounds for lambda_e")
        return delta, coeffs_of_ergod, erg_rate

    def long_term_uniform_error(self, m=0, n=50, delta_max=None, ax=None):
        """Approximate the long term error for the uniform-ergodic method.

        This method implements Proposition 9 of [1].

        Parameters
        ----------
        m : int, optional
            The number of repeated applications of (I + \delta \lowrateop). If
            equal to zero, then `dim - 1` is
            used.
        n : int, optional
            The number of deltas in between 0 and `delta_max`
        delta_max : float, optional
            The maximal value of delta. If `delta_max` is equal to `None`,
            then the maximally allowable value (`self.deltamax`) is used.
        ax : axis, optional
            A `pyplot` plot axis, if we want to plot the results

        Returns
        -------
        delta : ndarray
            A `ndarray` of shape `(1,n+1)` that contains all the delta's
            that were used for the approximation.
        coeffs_of_ergod : ndarray
            A `ndarray` of shape `(2, n+1)`; the first row contains the
            lower bounds for the long-term ergodic error, the second the
            upper bounds.
        """
        print("// Determining bounds for the long-term uniform error.")
        if m is 0:
            _m = self.dim - 1
        else:
            _m = m
        if delta_max is None:
            deltamax = self.deltamax
        else:
            deltamax = min(delta_max, self.deltamax)
        ergodic_error = np.empty((2, n+1))
        delta = np.linspace(0, deltamax, num=n+2)[1:]
        _normQ = self.norm
        _det_erg_coeff = self._determine_coeff_of_ergod
        # Determining the upper bound on the ergodic error
        for i in trange(n+1):
            coeffs = _det_erg_coeff(delta[i], rep=_m)
            # Lower bound
            beta = coeffs[0]
            ergodic_error[0, i] = _m * delta[i]**2 * _normQ ** 2 / (1-beta)
            # Upper bound
            beta = coeffs[1]
            ergodic_error[1, i] = _m * delta[i]**2 * _normQ ** 2 / (1-beta)
        # Check whether we need to plot the coefficients of ergodicity
        if ax is not None:
            ax.plot(delta, ergodic_error[0], '.-', label="Lower bound")
            ax.plot(delta, ergodic_error[1], '.-', label="Upper bound")
            ax.set_xlabel("delta")
            ax.set_ylabel("Bound for epsilon' / norm(f)_c with  m = {}".format(
                _m))
        return delta, ergodic_error

    def approx_conditional_lower_previsions(
            self, gamble, timeper, start=0, eps=1e-3, numplot=100,
            method='adaptive', maxiter=None, tightererror=False,
            itersteps=1):
        """Approximate a conditional lower prevision.

        This methods simply points to the correct method.

        Parameters
        ----------
        gamble : array_like
            The function of which the expectation is to be approximated.
        timeper : float
            The length of (t-s)
        start : float, optional
            The time instant s at which to start the plots.
        eps : float, optional
            The desired maximal error.
        numplot : int, optional
            The number of points used to plot the evolution of the
            approximation.
        method : {'adaptive, 'uniform', 'uniform-ergodic'}, optional
            The algorithm that is to be used
        maxiter : int, optional
            Overrides the number of iterations of the `'uniform'` method if not
            `None`.
        tightererror : bool, optional
            If `True`, then a tighter errorbound is determined, which increases
            the duration of the computations.
        itersteps : int, optional
            For the `'adaptive'` method, this is the number of iterations after
            which to re-evaluate the step size.
            For the `'uniform'` and `'uniform-ergodic'` methods, this is the
            parameter `m`.

        Returns
        -------
        ConditionalLowerPrevisionsApprox
            An approximation.
        """
        if self.norm is 0 or compute_centred_norm(gamble) is 0 or timeper is 0:
            # Approximation is not necessary
            approx = ConditionalLowerPrevisionsApprox(
                gamble, timeper, eps, start=start, isergodic=self.isergodic)
            approx._add_plot_step(timeper, gamble, 0, timeper, 0)
            approx._end_of_computations(0, actualerror=True)
            return approx
        elif method is 'adaptive':
            return self._approx_conditional_lower_previsions_adaptive(
                gamble, timeper, start=start, eps=eps, numplot=numplot,
                tightererror=tightererror, itersteps=itersteps)
        elif method is 'uniform':
            return self._approx_conditional_lower_previsions_uniform(
                gamble, timeper, start=start, eps=eps, numplot=numplot,
                maxiter=maxiter, tightererror=tightererror,
                itersteps=0, ergodic_decrease=False)
        elif method == 'uniform-ergodic':
            return self._approx_conditional_lower_previsions_uniform(
                gamble, timeper, start=start, eps=eps, numplot=numplot,
                maxiter=maxiter, tightererror=tightererror,
                itersteps=itersteps, ergodic_decrease=True)
        else:
            print("ERROR")

    def _approx_conditional_lower_previsions_uniform(
            self, gamble, timeper, start=0, eps=1e-3, numplot=100,
            inistep=None, maxiter=None, tightererror=False,
            itersteps=0, ergodic_decrease=True):
        """Approximate a conditional lower prevision with a uniform grid.

        This method implements Algorithm 1, Theorem 5 and Proposition 9 of [1].

        Parameters
        ----------
        gamble : array_like
            The function of which the expectation is to be approximated.
        timeper : float
            The length of (t-s)
        start : float, optional
            The time instant s at which to start the plots.
        eps : float, optional
            The desired maximal error.
        numplot : int, optional
            The number of points used to plot the evolution of the
            approximation.
        inistep : float, optional
            Overrides the initial step size if not `None`.
        maxiter : int, optional
            Overrides the number of iterations if not `None`.
        tightererror : bool, optional
            If `True`, then a tighter errorbound is determined, which increases
            the duration of the computations.
        itersteps : int, optional
            Only used for the `'uniform-ergodic'` method. The value of
            `itersteps` determines the value of the parameter m. If equal to 1,
            then `dim - 1` is used.
        ergodic_decrease : bool, optional
            If `True`, then the ergodic upper bound on the error is used to
            a priori decrease the number of iterations. Otherwise, the standard
            uniform method is used.

        Returns
        -------
        ConditionalLowerPrevisionsApprox
            An approximation.
        """
        approx = ConditionalLowerPrevisionsApprox(
            gamble, timeper, eps, start=start,
            isergodic=self.is_ergodic)
        # Time the duration of the computations.
        timing = time.time()
        # Determine the total number of iterations
        normF = compute_centred_norm(gamble)
        _normQ = self.norm
        min_n = math.ceil(timeper*_normQ/2)
        n = max(min_n, math.ceil(timeper**2*_normQ**2*normF/eps))
        # Corrections on numiter taking into account max_iteration
        if maxiter is not None:
            if maxiter < numplot - 1:
                numplot = maxiter + 1
                n = maxiter
            elif n > maxiter:
                n = maxiter
        elif inistep is not None:
            n = math.ceil(timeper/inistep)
        # Check whether or not we can actually increase the step size,
        # or in other words decrease the number of iterations
        if ergodic_decrease:
            # We first determine the ergodic upper bound on the error for the
            # initial number of iterations.
            _det_erg_coeff = self._determine_coeff_of_ergod
            opt_n = n
            if not itersteps:
                _m = self.dim - 1
            else:
                _m = itersteps
            k = math.ceil(opt_n / _m)
            delta = timeper / opt_n
            beta = _det_erg_coeff(delta, rep=_m)[1]
            opt_ergodic_error = (_m * delta**2 * _normQ**2
                                 * normF * (1 - beta**k) / (1 - beta))
            print("// We would initially use {:,} iterations.".format(n))
            print("// The ergodic bound for the error is {} ({})".format(
                opt_ergodic_error, opt_ergodic_error/(1-beta**k)))
            # This is a heuristic method to obtain a lower n, only used for the
            # 'uniform-ergodic' method
            if opt_ergodic_error < eps:
                new_n = max(opt_n//2, min_n)
                bad_n = 0
                i, max_it = 0, 20
                while i < max_it:
                    i += 1
                    k = math.ceil(new_n / itersteps)
                    delta = timeper / new_n
                    beta = _det_erg_coeff(delta, rep=_m)[1]
                    new_ergodic_error = (itersteps * delta**2 * _normQ**2
                                         * normF * (1 - beta**k) / (1 - beta))
                    if new_ergodic_error < eps:
                        # print("Better n is {:,}".format(new_n))
                        opt_n = new_n
                        opt_ergodic_error = new_ergodic_error
                        if bad_n is 0:
                            new_n = max(opt_n//2, min_n)
                        else:
                            new_n = max((bad_n+opt_n)//2, min_n)
                    else:
                        bad_n = new_n
                        new_n = max((opt_n + new_n)//2, min_n)
                    if new_n is opt_n:
                        i = max_it
                n = opt_n
                print(("// Due to ergodicity, we can reduce the number of"
                       "iterations to {:,}").format(n))
            else:
                print("// The number of iterations cannot be taken lower")
        approx.numiter = n
        timestep = timeper/n  # Small time step used in iterations
        approx._set_first_step(timestep)
        print("Number of total iterations is {:,}, with delta = {}.".format(
            approx.numiter, timestep))
        # Fields that are used for plotting
        timeinc = 0
        errorinc = 0
        numiter = 0
        ttol = .99999 * timeper / (numplot - 1)
        g = np.copy(gamble)
        # Actual computations
        _eval = self.eval
        for i in trange(n):
            # Determine the tighter error bound
            if tightererror:
                normG = compute_centred_norm(g)
                errorinc += _normQ**2 * timestep**2 * normG
            # Do the actual computation
            g = np.add(g, timestep * _eval(g))
            # Increment the fields for plotting
            timeinc += timestep
            numiter += 1
            # Check whether a plot point needs to be added.
            if timeinc >= ttol:
                approx._add_plot_step(
                    timeinc, g, errorinc, timestep, numiter)
                timeinc = 0
                errorinc = 0
        # It might be necessary to add a final plot step
        if timeinc > 0:
            approx._add_plot_step(
                    timeinc, g, errorinc, timestep, numiter)
        timing = time.time() - timing
        approx._end_of_computations(timing, actualerror=tightererror)
        return approx

    def _approx_conditional_lower_previsions_adaptive(
            self, gamble, timeper, start=0, eps=1e-3, numplot=100,
            tightererror=False, itersteps=1):
        """Approximate a conditional lower prevision with an adaptive grid.

        This method implements Algorithm 1 and Theorem 6 of [1].

        Parameters
        ----------
        gamble : array_like
            The function of which the expectation is to be approximated.
        timeper : float
            The length of (t-s)
        start : float, optional
            The time instant s at which to start the plots.
        eps : float, optional
            The desired maximal error.
        numplot : int, optional
            The number of points used to plot the evolution of the
            approximation.
        tightererror : bool, optional
            If `True`, then a tighter errorbound is determined, which increases
            the duration of the computations.
        itersteps : int, optional
            Determines the number of iterations after which the step-size
            should be re-evaluated. If equal to `0`, then `dim - 1` is used.

        Returns
        -------
        ConditionalLowerPrevisionsApprox
            An approximation.
        """
        approx = ConditionalLowerPrevisionsApprox(
            gamble, timeper, eps, start=start,
            isergodic=self.is_ergodic)
        # Time the duration of the computations.
        timing = time.time()
        # Determine the total number of iterations
        normF = compute_centred_norm(gamble)
        _normQ = self.norm
        timestep = min(eps / (timeper * _normQ**2 * normF), timeper, 2/_normQ)
        print("Maximum number of iterations is {:,}, with initial delta = {}."
              .format(math.floor(timeper/timestep), timestep))

        g = np.copy(gamble)
        normG = compute_centred_norm(g)
        _deltamax = self.deltamax
        pbartotal = 1000
        with tqdm(total=pbartotal) as pbar:
            # Variables used for the progress bar
            barval = 0
            # Variables necessary for the algorithm
            sumprevdelta = 0
            cumerror = 0
            if itersteps:
                _m = min(itersteps, math.ceil(timeper / timestep))
            else:
                _m = min(self.dim - 1, math.ceil(timeper / timestep))
            # Variables used for plotting
            timeinc = 0
            errorinc = 0
            numiter = 0
            ttol = .9999 * timeper / (numplot - 1)
            timestep = min(eps / (timeper * _normQ**2 * normG),
                           timeper, _deltamax)
            approx._set_first_step(timestep)
            # Actual computations
            _eval = self.eval
            while sumprevdelta < timeper:
                    normG = compute_centred_norm(g)
                    if normG is 0:
                        # We need to stop the computations
                        timestep = timeper - sumprevdelta
                        timeinc += timestep
                        break
                    timestep = min(
                        eps / (timeper * _normQ**2 * normG),
                        timeper - sumprevdelta, _deltamax)
                    if timestep * _m > timeper - sumprevdelta:
                        _m = math.ceil((timeper - sumprevdelta)
                                       / timestep)
                        timestep = (timeper - sumprevdelta) / _m
                    if not tightererror:
                        # Then we keep track of this upper bound
                        localerror = (_m * _normQ**2 * timestep**2
                                      * normG)
                        errorinc += localerror
                        cumerror += localerror
                    # Do _m steps
                    for i in range(_m):
                        if tightererror:
                            # We determine the local error after each iteration
                            normG = compute_centred_norm(g)
                            localerror = _normQ**2 * timestep**2 * normG
                            errorinc += localerror
                            cumerror += localerror
                        g = np.add(g, timestep * _eval(g))
                    # Increment stuff for the plots
                    timeinc += _m * timestep
                    sumprevdelta += _m * timestep
                    numiter += _m
                    # Check whether a plot point needs to be added.
                    if timeinc >= ttol:
                        approx._add_plot_step(
                            timeinc, g, errorinc, timestep, numiter)
                        timeinc = 0
                        errorinc = 0
                    # Finally, we check whether the progressbar needs updating.
                    newbarval = math.floor(pbartotal*sumprevdelta/timeper)
                    if newbarval-barval >= pbartotal/100:
                        pbar.update(newbarval - barval)
                        barval = newbarval
        # It might be necessary to add a final plot step
        if timeinc > 0:
            approx._add_plot_step(
                    timeinc, g, errorinc, timestep, numiter)
        # Finalise the approx object
        timing = time.time() - timing
        approx._end_of_computations(timing, actualerror=True)
        return approx

    def approx_limit_lower_previsions(
            self, gamble, eps=1e-3, itersteps=None, ini_delta=None, div=5):
        """Approximate the limit value of the conditional lower prevision.

        This method implements Proposition 12 of [1].

        Parameters
        ----------
        gamble : array_like
            The function of which the expectation is to be approximated.
        eps : float, optional
            The desired maximal error.
        itersteps : int, optional
            Determines the number of iterations after which the step-size
            should be re-evaluated. If equal to `0`, then `dim - 1` is used.
        ini_delta : float, optional
            An initial value of delta.
        div : int, optional
            The number of divisions to be used in the heuristic to find the
            step size.

        Returns
        -------
        ndarray
            The final function of our approximation method. The centerpoint of
            this function can be used as approximation for the limit value.
        """
        # Determine the necessary m and delta
        if not itersteps:
            _m = self.dim - 1
        else:
            _m = itersteps
        normF = compute_centred_norm(gamble)
        _normQ = self.norm
        if not ini_delta:
            delta = self.deltamax/10
        else:
            delta = min(ini_delta, self.deltamax)
        _det_erg_coeff = self._determine_coeff_of_ergod
        beta = _det_erg_coeff(delta, rep=_m)[1]
        ergodic_error = (_m * delta**2 * _normQ**2
                         * normF / (1 - beta))
        if ergodic_error < eps / 2:
            opt_delta = delta
            opt_ergodic_error = ergodic_error
            bad_delta = self.deltamax
        else:
            opt_delta = 0
            bad_delta = delta
            opt_ergodic_error = eps
        i, max_it = 0, 20
        while i < max_it:
            i += 1
            found = False
            for j in range(div-1, 0, -1):
                delta = opt_delta + j * (bad_delta - opt_delta) / div
                beta = _det_erg_coeff(delta, rep=_m)[1]
                ergodic_error = (_m * delta**2 * _normQ**2
                                 * normF / (1 - beta))
                if ergodic_error < eps / 2:
                    opt_delta = delta
                    bad_delta = opt_delta + (j+1)*(bad_delta - opt_delta)/div
                    opt_ergodic_error = ergodic_error
                    found = True
                    break
            if not found:
                bad_delta = opt_delta + (bad_delta - opt_delta)/div
        if opt_ergodic_error < eps / 2:
            delta = opt_delta
            print("// The optimal delta is {}".format(delta))
        else:
            print("// No optimal delta was found")
            return None
        # Actual computations
        g = np.copy(gamble)
        normG = normF
        # compeps = 0
        numiter = 0
        _eval = self.eval
        print("// Running computations, might take a while.")
        pbartotal = 10000
        pbar = tqdm(total=pbartotal)
        barval = 0
        while 2 * normG > eps:
            for i in range(_m):
                g = np.add(g, delta*_eval(g))
            numiter += _m
            normG = compute_centred_norm(g)
            fact = (2 * normG - eps) / (2 * normF - eps)
            newbarval = pbartotal * (1 - fact)
            if newbarval - barval > 1:
                diff = math.floor(newbarval - barval)
                pbar.update(diff)
                barval += diff
        pbar.close()
        print("After {:,} iterations, we have found that".format(numiter))
        print("The limit value is = {} +- {}".format(np.average(g), eps))
        return g

    def approx_limit_lower_previsions_alt(
            self, gamble, delta, eps=1e-3, tightererror=False, itersteps=None):
        """Approximate the limit value of a conditional lower prevision.

        This method implements the alternative method discussed in Section 5.4
        of [1].

        Parameters
        ----------
        gamble : array_like
            The function of which the expectation is to be approximated.
        delta : float
            The step size to be used.
        eps : float, optional
            The desired maximal error.
        tightererror : bool, optioanl
            If `True`, then a tighter error is used.
        itersteps : int, optional
            The number of iterations after which to check whether or not the
            approximation has sufficiently low norm. If `None`, then `dim` is
            used.

        Returns
        -------
        ndarray
            The final function of our approximation method. The centerpoint of
            this function can be used as approximation for the limit value.
        """
        g = np.copy(gamble)
        normG = compute_centred_norm(g)
        _normQsq = self.norm**2
        _eval = self.eval
        if itersteps is None:
            _itersteps = self.dim
        else:
            _itersteps = itersteps
        eps_fact = delta**2 * _normQsq
        eps = 0
        numloop = 1
        pbartotal = 10000
        pbar = tqdm(total=pbartotal)
        barval = 0
        while normG > eps:
            for i in range(_itersteps):
                if tightererror:
                    normG = compute_centred_norm(g)
                    eps += eps_fact * normG
                g = np.add(g, delta * _eval(g))
            if not tightererror:
                eps += _itersteps * eps_fact * normG
                normG = compute_centred_norm(g)
            numloop += 1
            fact = (normG - eps) / normG
            newbarval = pbartotal * (1 - fact)
            if newbarval - barval > 1:
                diff = math.floor(newbarval - barval)
                pbar.update(diff)
                barval += diff
        pbar.close()
        print("We needed {:,} iterations, but have found that".format(
            numloop*_itersteps))
        print("limit = {} +- {}".format(np.average(g), 2*eps))
        return g, 2*eps


class BinaryLowerTransitionRateOperator(LowerTransitionRateOperator):
    """A lower transition rate operator with binary state space.

    This class extends `LowerTransitionRateOperator`, and contains analytical
    methods to check for ergodicity, determine the coefficient of ergodicity
    of the approximation. Also, it contains a method to exactly determine the
    actual value of a conditional lower expectation, and another one to
    exactly determine the limit value.
    """
    def __init__(self, lambmat):
        """Initialise a `BinaryLowerTransitionRateOperator`.

        Parameters
        ----------
        lambmat : ndarray
            An `ndarray` with shape (2,2). The first row contains the interval
            q_0, the second row contains the interval q_1.
        """
        self.dim = 2
        self._lambdamat = lambmat
        self._determine_ergodic_rate()
        self.is_ergodic = self.check_ergodicity()
        self.norm = 2 * max(lambmat[:, 1])
        print("// Initialised a binary lower transition rate operator")
        print("// Norm of Q = {}, lambda_e = {}".format(self.norm,
              self.ergodic_rate))
        if self.norm > 0:
            self.deltamax = 2 / self.norm
        else:
            self.deltamax = math.inf

    def eval(self, gamble):
        """Evaluate the binary lower transition rate operator.

        Parameters
        ----------
        gamble : array_like
            The gamble f that you want to apply the lower transition rate
            operator to.

        Returns
        -------
        ndarray
            An `ndarray` of shape (1, 2) that is equal to Q f.
        """
        try:
            if gamble.size != self.dim:
                raise ValueError('dim of gamble != dim of state space')
            else:
                ret = np.empty(2)
                diff = gamble[0] - gamble[1]
                if diff > 0:
                    ret[0] = - self._lambdamat[0, 1] * diff
                    ret[1] = self._lambdamat[1, 0] * diff
                else:
                    ret[0] = - self._lambdamat[0, 0] * diff
                    ret[1] = self._lambdamat[1, 1] * diff
                return ret
        except ValueError as error:
            print(error)
            return gamble

    def compute_lower_rate_norm(self):
        """Determine the norm of the binary lower transition rate operator.

        Returns
        -------
        float
            The maximum norm of the binary lower transition rate operator.
        """
        self.norm = 2 * max(self._lambdamat[:, 1])
        return self.norm

    def _determine_ergodic_rate(self):
        """Determine the ergodic rate.

        Returns
        -------
        float
            The ergodic rated associated with the binary lower transition rate
            operator.
        """
        self.ergodic_rate = min(self._lambdamat[0, 1] + self._lambdamat[1, 0],
                                self._lambdamat[0, 0] + self._lambdamat[1, 1])
        return self.ergodic_rate

    def alt_check_ergodicity(self):
        self._determine_ergodic_rate()
        if self.ergodic_rate > 0:
            return True
        else:
            return False

    def eval_erg_function(self, t):
        """Evaluate the ergodic function in delta.

        Parameters
        ----------
        t : float
            A time point t.

        Returns
        -------
        float
            The ergodic function associated with the binary lower transition
            rate operator, evaluated inn the time point t.
        """
        return np.exp(- self.ergodic_rate * t)

    def _determine_coeff_of_ergod(self, delta, rep=1):
        """Determine the coefficient of ergodicity of the approximation.

        Parameters
        ----------
        delta : float
            A small time step delta.
        rep : int, optional
            The factor m.

        Returns
        -------
        coeff_erg : list of floats
            Both of the elements of `coeff_erg` are the exact value of
            rho((I + \delta Q)^{m}).
        """
        l1 = (self._lambdamat[0, 1] + self._lambdamat[1, 0])
        l2 = (self._lambdamat[0, 0] + self._lambdamat[1, 1])
        if delta <= 1 / l1:
            c1 = (1 - delta * l1)**rep
            if delta <= 1 / l2:
                c2 = (1 - delta * l2)**rep
            else:
                c2 = (1 - delta * l1)**(rep-1) * (1 - delta * l2)
        else:
            if delta <= 1 / l2:
                c1 = (1 - delta * l1) * (1 - delta * l2)**(rep-1)
                c2 = (1 - delta * l2)**rep
            else:
                i = math.ceil(rep/2)
                j = math.floor(rep/2)
                c1 = (1 - delta * l1)**i * (1 - delta * l2)**j
                c2 = (1 - delta * l1)**j * (1 - delta * l2)**i
        coeff_erg = max(abs(c1), abs(c2))
        coeff_erg = [coeff_erg, coeff_erg]  # For compatibility
        return coeff_erg

    def actual_conditional_lower_prevision(
            self, gamble, timeper):
        """Compute the actual value of the conditional lower prevision.

        Parameters
        ----------
        gamble : array_like
            The function of which we would like to know the conditional lower
            prevision.
        timeper : float
            The length of the time period (t-s).

        Returns
        -------
        lprev : ndarray
            A `ndarray` with shape (1, 2) that contains the conditional lower
            prevision.
        """
        try:
            if gamble.size is not self.dim:
                raise ValueError("""Dimension of gamble and state space are
                                 different.""")
            else:
                _lambmat = self._lambdamat
                lprev = np.empty(2)
                diff = gamble[0] - gamble[1]
                if diff > 0:
                    lam = (_lambmat[0, 1] + _lambmat[1, 0])
                    factor = (1 - np.exp(- timeper * lam)) / lam
                    lprev[0] = gamble[0] - diff * _lambmat[0, 1] * factor
                    lprev[1] = gamble[1] + diff * _lambmat[1, 0] * factor
                elif diff is 0:
                    lprev = np.copy(gamble)
                else:
                    lam = (_lambmat[0, 0] + _lambmat[1, 1])
                    factor = (1 - np.exp(- timeper * lam)) / lam
                    lprev[0] = gamble[0] - diff * _lambmat[0, 0] * factor
                    lprev[1] = gamble[1] + diff * _lambmat[1, 1] * factor
                return lprev
        except ValueError as error:
            print(error)
            return gamble

    def actual_limit_lower_prevision(self, gamble):
        """Compute the limit value of the conditional lower prevision.

        Parameters
        ----------
        gamble : array_like
            The function of which we would like to know the limit of its
            conditional lower prevision.

        Returns
        -------
        float
            The limit value of the conditional lower prevision of the
            supplied function.
        """
        _lambmat = self._lambdamat
        try:
            if gamble.size is not self.dim:
                raise ValueError("""Dimension of gamble and state space are
                                 different.""")
            else:
                diff = gamble[0] - gamble[1]
                if diff > 0:
                    factor = _lambmat[0, 1] / (_lambmat[0, 1] + _lambmat[1, 0])
                else:
                    factor = _lambmat[0, 0] / (_lambmat[0, 0] + _lambmat[1, 1])
                return gamble[0] - diff * factor
        except ValueError as error:
            print(error)
            return gamble


class ConditionalLowerPrevisionsApprox:
    """A class to study the evolution of approximative methods over time.

    The `ConditionalLowerPrevisionsApprox` class contains methods to plot the
    time-evolution, as well as print out the final approximation to the
    terminal.
    """
    def __init__(self, gamble, timeper, max_error, start=0, isergodic=False):
        """Initialise a `ConditionalLowerPrevisionsApprox` object.

        Parameters
        ----------
        gamble : array_like
            The function f of which the lower prevision is approximated.
        timeper : float
            The time period (t-s).
        max_error : float
            The desired maximal error of the approximation.
        start : float, optional
            The starting time s of the approximation, used for plotting.
        is_ergodic : bool, optional
            If `True`, then something about the ergodic convergence is said.
        """
        self.dim = gamble.size
        self.timeper = timeper
        self.finaltime = start + timeper
        self.is_ergodic = isergodic
        self.t_plot = [start]
        self.lowerprevisions_plot = [np.array(gamble)]
        self.timestep_plot = [0]
        self.numiter_plot = [0]
        self.error_plot = [0]
        self.value = None
        self.numiter = None
        self.max_error = max_error
        self.lower_error = None
        self.compduration = None

    def _add_plot_step(
            self, tinc, lowprev, errinc, iterstep, niter):
        """Add a plot step.

        Parameters
        ----------
        tinc : float
            Time incerement
        lowprev : array_like
            Current value of the conditional lower previsions.
        errinc : float
            Error increment.
        iterstep : float
            Current value of the step size delta
        niter : int
            Cumulative number of iterations

        """
        self.t_plot.append(self.t_plot[-1] + tinc)
        self.lowerprevisions_plot.append(lowprev)
        self.error_plot.append(self.error_plot[-1] + errinc)
        self.timestep_plot.append(iterstep)
        self.numiter_plot.append(niter)

    def _set_first_step(self, delta):
        """Set the initial value of the step size.

        Parameters
        ----------
        delta : float
            Initial value of the step size delta
        """
        self.timestep_plot[0] = delta

    def _end_of_computations(self, duration, actualerror=False):
        """Obtain end values of relevant parameters.

        Parameters
        ----------
        duration : float
            Computation of duration
        actualerror : bool, optional
            Set to `True` if a lower bound for the error was being determined.
        """
        self.numiter = self.numiter_plot[-1]
        self.lowerprevisions_plot = np.array(self.lowerprevisions_plot)
        self.value = self.lowerprevisions_plot[-1]
        if actualerror:
            self.lower_error = self.error_plot[-1]
        else:
            self.error_plot = [self.max_error * i / self.numiter
                               for i in self.numiter_plot]
        self.compduration = duration

    def final_approx(self):
        return self.value

    def total_duration(self):
        return self.compduration

    def final_error(self):
        if self.lower_eror is not None:
            return self.lower_error
        else:
            return self.max_error

    def plot_lowerprevision(self, ax=plt.gcf(), prefix=""):
        """Plot the evolution of the approximation.

        Parameters
        ----------
        ax : axis, optional
            Axis to plot the evolution in. Default value is the current figure.
        prefix : str, optional
            A string to prepend the labels with.
        """
        c = cm.viridis(np.linspace(0, 1, num=self.dim))
        for i in range(0, self.dim):
            ax.fill_between(
                self.t_plot, self.lowerprevisions_plot[:, i]-self.error_plot,
                self.lowerprevisions_plot[:, i]+self.error_plot,
                color=c[i], alpha=.2)
            ax.plot(self.t_plot, self.lowerprevisions_plot[:, i], '.-',
                    color=c[i], label=prefix+"X_t = {}".format(i))

    def plot_numiter(self, ax=plt.gcf(), c='k', label=None):
        """Plot the evolution of the cumulative number of iterations.

        Parameters
        ----------
        ax : axis, optional
            Axis to plot the evolution in. Default value is the current figure.
        c : str, optional
            Color format
        label : str, optional
            Add a label to the plot.
        """
        ax.plot(self.t_plot, self.numiter_plot, '.-', color=c, label=label)

    def plot_timestep(self, ax=plt.gcf(), c='k', label=None):
        """Plot the evolution of the step size delta.

        Parameters
        ----------
        ax : axis, optional
            Axis to plot the evolution in. Default value is the current figure.
        c : str, optional
            Color format
        label : str, optional
            Add a label to the plot.
        """
        ax.plot(self.t_plot, self.timestep_plot, '.-', color=c, label=label)

    def print_final_approx(self):
        """Print the final approximation to the terminal."""
        timediff = datetime.timedelta(seconds=self.compduration)
        print("{:,} iterations in {}".format(self.numiter,
                                             timediff))
        print("The approximation is:")
        print(self.value)
        if self.lower_error is not None:
            print("A certain upper bound for the error is {}"
                  .format(self.lower_error))
        if self.is_ergodic:
            cent_norm = compute_centred_norm(self.value)
            if cent_norm < self.max_error:
                cent_value = (max(self.value) + min(self.value)) / 2
                print(("Convergence has occurred,"
                       " the limit of the prevision is {}")
                      .format(cent_value))
            else:
                print("Convergence has not occurred.")
