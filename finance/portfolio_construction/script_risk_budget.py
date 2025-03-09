import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

import finance.portfolio_construction.scipy_port_constraints as pc

'''
Sample script by kenchan323 (https://github.com/kenchan323) to perform portfolio risk budgeting optimisation under two
different approaches (convex and non-convex). A basic dummy 4 by 4 covariance matrix is used and the script shows that
the convex approach (solved using cvxpy) is more optimal than the non-convex approach (solved using scipy solver).
However, the latter is more flexible for introducing more sophisticated constraints (beyond just long-only, weights
sum to 1 etc).

@kenchan323
2020-09-20

References:
The Quant MBA (DZ)
https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/
Fast Design of Risk Parity Portfolios (Zé Vinícius & Daniel P. Palomar)
https://cran.r-project.org/web/packages/riskParityPortfolio/vignettes/RiskParityPortfolio.html#using-the-ackage-riskparityportfolio
'''

def solve_convex_risk_budget_obj_func(cov, w_target):
    '''
    This is to implement the obj function under the "Vanilla convex formulation" section in the below page:
    https://cran.r-project.org/web/packages/riskParityPortfolio/vignettes/RiskParityPortfolio.html#using-the-
    package-riskparityportfolio
    This convex problem can only fulfil basic constraints like long only. Optimal solution is guaranteed here because
    of convex form
    If more sophisticated constraints are needed then use the non-convex version

    :param cov: 2d list - n by n covariance matrix
    :param w_target: list - n by 1 for target risk contribution (e.g. [0.25, 0.25, 0.25, 0.25] if n == 4 and you want
    equal risk parity solution)
    :return: list - optimal weights
    '''
    w_target = np.asmatrix(w_target)
    n = w_target.shape[1]
    w = cp.Variable((n, 1))
    a = 0.5 * cp.quad_form(w, cov)
    b = w_target @ cp.log(w)
    cp_prob = cp.Problem(cp.Minimize(a-b),
                         constraints=[w >= 0])
    cp_prob.solve()
    x_conv_res = [x[0] for x in w.value]
    # Need to convert back to weight (see the link)
    x_conv_res = [x / sum(x_conv_res) for x in x_conv_res]
    return x_conv_res


def _non_convex_risk_budget_objective(x, cov, x_target):
    '''
    This is the implementation of the NON-convex objective function for solving a risk budgeting solution in the below
    page:
    https://thequantmba.wordpress.com/2016/12/14/risk-parityrisk-budgeting-portfolio-in-python/
    :param x: list - n by 1 variable
    :param cov: 2d list - n by n covariance matrix
    :param x_target: list - n by 1 for target risk contribution (e.g. [0.25, 0.25, 0.25, 0.25] if n == 4 and you want
    equal risk parity solution)
    :return: float - evaluated value of the objective function
    '''

    def _calculate_risk_contribution(w, cov):
        '''
        Calculate the fractions of risk contribution of a list of assets
        :param w: list - portfolio weights
        :param cov: 2d list - n by n covariance matrix
        :return: list - fractions of risk contribution
        '''
        # function that calculates asset contribution to total risk
        w = np.asmatrix(w)
        port_variance = (w*cov*w.T)[0,0]
        port_sigma = np.sqrt(port_variance)
        # Marginal Risk Contribution
        mrc = cov * w.T
        # Risk Contribution
        risk_contr = np.multiply(mrc, w.T) / port_sigma
        return risk_contr
    x = np.asmatrix(x)
    cov = np.asmatrix(cov) # covariance matrix
    port_var = (x@cov@x.T)[0,0] # portfolio variance
    port_sigma =  np.sqrt(port_var) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(port_sigma,x_target))
    asset_rc = _calculate_risk_contribution(x,cov)
    sse = sum(np.square(asset_rc - risk_target.T))[0,0] # sum of squared error
    return sse




def _risk_budget_obj_failed_attempt(cov, w_target):
    '''
    This was an uneducated/failed attempt to formulate the non-convex problem using CVXPY (didn't work as it's not
    a convex problem, hence doesn't follow cvxpy's DCP rules)
    '''
    # This straight forward case is non-convex!
    n = np.asmatrix(cov).shape[0]
    w = cp.Variable((n, 1))
    sig_p = cp.quad_form(w,cov)
    #print("sig_p " + str(sig_p.shape))
    risk_target = sig_p*w_target
    #print("risk_target " + str(risk_target.shape))
    # asset_RC = calculate_risk_contribution(w, cov)
    mrc = cov@w
    #print("mrc " + str(mrc.shape))
    #print("cp.multiply(mrc,w) " + str(cp.multiply(mrc,w).shape))
    asset_RC = cp.multiply(mrc,w)/sig_p # asset_RC = ((4,1) * (1,4)) / (1,1)
    #print("asset_RC " + str(asset_RC.shape))
    return cp.Problem(cp.Minimize(cp.sum_squares(asset_RC - risk_target)),
                      constraints=[cp.sum(w) == 1,
                                   w >= 0])

# Some dummy data
x_target = [0.25, 0.25, 0.25, 0.25] # target risk budgeting distribution (here implying ERP)
w_0 = [0.25, 0.25, 0.25, 0.25] # initial guesses
# dummy 4 by 4 covariance matrix
cov = [[1.23, 0.375, 0.7, 0.3],
       [0.375, 1.22, 0.72, 0.135],
       [0.7, 0.72, 3.21, -0.32],
       [0.3, 0.135, -0.32, 0.52]]

# 1) Let's solve it as a non-convex problem (using scipy solver)
cons = ({'type': 'eq', 'fun': pc.total_weight_constraint},
        {'type': 'ineq', 'fun': pc.long_only_constraint})
res = minimize(_non_convex_risk_budget_objective, w_0, args=(cov, x_target),
               method='SLSQP', constraints=cons, options={'disp': True})
w_sol_non_convex = res.x

# 2) Structure as and solve as a convex problem (using cvxpy)
w_sol_convex = solve_convex_risk_budget_obj_func(cov, x_target)

# We evaluate the (non-convex) objective functions using the two sets of solutions
eval_obj_fuc_non_convex = _non_convex_risk_budget_objective(w_sol_non_convex, cov, x_target)
eval_obj_fuc_convex = _non_convex_risk_budget_objective(w_sol_convex, cov, x_target)

print("Solution to the non-convex problem:")
print([np.round(x, 5) for x in w_sol_non_convex])
print("Solution to the convex problem:")
print([np.round(x, 5) for x in w_sol_convex])

assert eval_obj_fuc_convex < eval_obj_fuc_non_convex
print("Evaluated with convex problem optimal solution < Evaluated with non-convex problem optimal solution")
print(f"{eval_obj_fuc_convex} < {eval_obj_fuc_non_convex}")
print("Solution to convex problem is MORE optimal than that to the non-convex problem")

