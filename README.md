# ictmc
A Python 3 module to approximate (conditional) lower expectations for imprecise continuous-time Markov chains

## About
This Python 3 module contains naive, non-optimised implementations of the algorithms proposed by De Bock (2017) and Erreygers and De Bock (2017).
The main reason for writing this module was to validate the results obtained in (Erreygers and De Bock, 2017), but I believe it can be used for general purposes as well.

The module imports [NumPy](http://www.numpy.org/) for easy vector manipulations, [matplotlib](http://matplotlib.org/) for plotting the approximations and [tqdm](https://github.com/tqdm/tqdm) to display nice progress bars.
All three of these packages should be installed in order for the package to work. 

## Examples
In [HealthySick.py](HealthySick.py), the lower transition rate operator introduced in Example 18 of (Krak et al., 2016) is used to showcase the differences between the algorithms.

We do the same in [ReliabilityOfPowerNetworks.py](ReliabilityOfPowerNetworks.py), but for the lower transition rate operator introduced in (Troffaes et al., 2015).
The linear programming problems are solved using the [`linprog`](http://cvxopt.org/userguide/coneprog.html#linear-programming) method of [CVXOPT](http://cvxopt.org/), which is a Python wrapper for [GLPK](https://www.gnu.org/software/glpk/).
If you cannot get this to work, you can resort to the [`linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) method of [SciPy](https://www.scipy.org/).
However, I would advice you not to do this, as this method is terribly slow.

## References
1. Jasper De Bock. _''The Limit Behaviour of Imprecise Continuous-Time Markov Chains''_.  Journal of Nonlinear Science (2017) 27:159 . [doi:10.1007/s00332-016-9328-3](http://dx.doi.org/10.1007/s00332-016-9328-3)
2. Alexander Erreygers and Jasper De Bock. _''Imprecise continuous-time Markov Chains: Appriomxation Methods and Ergodicity''_. Work in progress.
3. Thomas Krak, Jasper De Bock and Arno Siebes. _''Imprecise Continuous-Time Markov Chains''_. [arXiv:1611.05796](https://arxiv.org/abs/1611.05796)
4. Matthias Troffaes, Jacob Gledhill, Damjan Skulj and Simon Blake. _''Using imprecise continuous time Markov chains for assessing the reliability of power networks with common cause failure and non-immediate repair''_. Proceedings of ISIPTA'15, pp. 287-294. [Available on-line](http://www.sipta.org/isipta15/data/paper/18.pdf)
