import warnings
import numpy as np
from scipy.optimize import minimize

import finance.portfolio_construction.scipy_port_constraints as pc

'''
Sample script to solve for a MDP (maximum diversification portfolio). Very similar to the implementation under the 
referenced link below, but the implementation of the long only constraint is slightly different 

@kenchan323
2020-12-27

References:
https://thequantmba.wordpress.com/2017/06/06/max-diversification-in-python/
'''
def _calc_diversification_ratio(w, cov):
    '''
    Calculate the diversification ratio of a portfolio. The return value is a negative signed one because this function
    is to be used as the objective function of a minimisation problem in solving for a MDP portfolio
    :param w: w_0: list - weights
    :param cov: 2d list - covariance matrix
    :return: float - negative of diversification ratio
    '''
    w = np.asmatrix(w)
    cov = np.asmatrix(cov)
    vol = np.diagonal(cov)
    # numerator
    sum_weighted_vol = np.sum(np.multiply(w, vol))
    # denominator
    port_vol = np.sqrt((w * cov * w.T)[0, 0])
    # Negatively signed as we trying a minimisation problem (minimise negative == maximise)
    return -sum_weighted_vol/port_vol


def solve_mdp_weights(w_0, cov, bnd=None, long_only=True):
    '''
    Solve for a maximum diversification portfolio (MDP)

    :param w_0: list - initial guess
    :param cov: 2d list - covariance matrix
    :param bnd: list of tuple - specific min/max bounds for each weight e.g.[(min_0,max_0),(min_1,max_1)..(min_n,max_n)]
    :param long_only: boolean - apply long only constraints or not
    :return: list - solution weights
    '''
    cons = ({'type': 'eq', 'fun': pc.total_weight_constraint},)
    if long_only: # add in long only constraint
        # Some caveat to this first approach and you may get a weight that is very mildly negative
        # constraints are not stable in numerical optimisation (strange!)
        # See the response:
        # https://stackoverflow.com/questions/45697017/scipy-optimizer-ignores-one-of-the-constraints
        # cons = cons + ({'type': 'ineq', 'fun':  lambda x: x},)

        if bnd is None: # instead we can make sure this is indicated via the bounds arg into minimize
            bnd = [(0, np.inf) for x in range(len(w_0))]
        else: # the caller has specified bounds!
            bnd_original = bnd.copy()
            # just make sure all the min's are at least greater than zero
            bnd = [(max(0, x[0]), x[1]) for x in bnd]
            if bnd != bnd_original:
                warnings.warn("As caller has indicated the need for a long-only solution, the updated bounds are:")
                print(bnd)

    res = minimize(_calc_diversification_ratio, w_0, bounds=bnd, args=cov,
                   method='SLSQP', constraints=cons)
    return [w_opt for w_opt in res.x]

if __name__ == 'main':
    w_0 = [0.25, 0.25, 0.25, 0.25] # initial guesses

    # dummy 4 by 4 covariance matrix
    cov = [[1.23, 0.375, 0.7, 0.3],
           [0.375, 1.22, 0.72, 0.135],
           [0.7, 0.72, 3.21, -0.32],
           [0.3, 0.135, -0.32, 0.52]]

    # Say I specify some bounds as I don't want a portfolio that is too concentrated (any stock to have > 50% weight)
    solution_mdp = solve_mdp_weights(w_0, cov,
                                     bnd=[(0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5)],
                                     long_only=True)


