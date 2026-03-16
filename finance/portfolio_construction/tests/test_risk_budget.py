import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Check if cvxpy is available
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

# Only import if cvxpy is available
if CVXPY_AVAILABLE:
    from finance.portfolio_construction import script_risk_budget as rb


@unittest.skipIf(not CVXPY_AVAILABLE, "cvxpy not installed")
class TestSolveConvexRiskBudget(unittest.TestCase):
    """Test cases for solve_convex_risk_budget_obj_func function"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov_4x4 = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

    def test_equal_risk_parity_4x4(self):
        """Test equal risk parity solution with 4x4 matrix"""
        w_target = [0.25, 0.25, 0.25, 0.25]
        result = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, w_target)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check all weights are non-negative
        self.assertTrue(all(w >= 0 for w in result))

        # Check correct number of weights
        self.assertEqual(len(result), 4)

    def test_unequal_risk_targets(self):
        """Test with unequal risk targets"""
        w_target = [0.4, 0.3, 0.2, 0.1]
        result = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, w_target)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check all weights are non-negative
        self.assertTrue(all(w >= 0 for w in result))

    def test_concentrated_risk_target(self):
        """Test with concentrated risk on one asset"""
        w_target = [0.7, 0.1, 0.1, 0.1]
        result = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, w_target)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # First weight should be larger
        self.assertGreater(result[0], result[1])

    def test_2x2_matrix(self):
        """Test with 2x2 covariance matrix"""
        cov_2x2 = np.array([
            [1.0, 0.5],
            [0.5, 2.0]
        ])
        w_target = [0.5, 0.5]

        result = rb.solve_convex_risk_budget_obj_func(cov_2x2, w_target)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check all weights are non-negative
        self.assertTrue(all(w >= 0 for w in result))

    def test_diagonal_covariance(self):
        """Test with diagonal (uncorrelated) covariance matrix"""
        cov_diag = np.diag([1.0, 2.0, 3.0, 4.0])
        w_target = [0.25, 0.25, 0.25, 0.25]

        result = rb.solve_convex_risk_budget_obj_func(cov_diag, w_target)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # For uncorrelated assets with equal risk targets,
        # weights should be inversely proportional to volatility
        # Higher variance assets should have lower weights
        for i in range(len(result) - 1):
            self.assertGreater(result[i], result[i + 1] - 1e-6)


@unittest.skipIf(not CVXPY_AVAILABLE, "cvxpy not installed")
class TestNonConvexRiskBudgetObjective(unittest.TestCase):
    """Test cases for _non_convex_risk_budget_objective function"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov_4x4 = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

    def test_objective_returns_scalar(self):
        """Test that objective function returns a scalar value"""
        x = [0.25, 0.25, 0.25, 0.25]
        x_target = [0.25, 0.25, 0.25, 0.25]

        result = rb._non_convex_risk_budget_objective(x, self.cov_4x4, x_target)

        self.assertIsInstance(result, (int, float, np.number))

    def test_objective_non_negative(self):
        """Test that objective function returns non-negative value"""
        x = [0.25, 0.25, 0.25, 0.25]
        x_target = [0.25, 0.25, 0.25, 0.25]

        result = rb._non_convex_risk_budget_objective(x, self.cov_4x4, x_target)

        # SSE should be non-negative
        self.assertGreaterEqual(result, 0)

    def test_perfect_solution_gives_zero(self):
        """Test that perfect risk parity solution gives objective close to zero"""
        # Get optimal solution from convex formulation
        x_target = [0.25, 0.25, 0.25, 0.25]
        x_optimal = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, x_target)

        # Evaluate objective with optimal solution
        result = rb._non_convex_risk_budget_objective(x_optimal, self.cov_4x4, x_target)

        # Should be close to zero
        self.assertLess(result, 1e-4)

    def test_equal_weights_non_optimal(self):
        """Test that equal weights are not optimal for risk parity"""
        x_equal = [0.25, 0.25, 0.25, 0.25]
        x_target = [0.25, 0.25, 0.25, 0.25]

        # Get optimal solution
        x_optimal = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, x_target)

        obj_equal = rb._non_convex_risk_budget_objective(x_equal, self.cov_4x4, x_target)
        obj_optimal = rb._non_convex_risk_budget_objective(x_optimal, self.cov_4x4, x_target)

        # Optimal should have lower objective value
        self.assertLess(obj_optimal, obj_equal)

    def test_different_target_distributions(self):
        """Test objective with different target distributions"""
        x = [0.25, 0.25, 0.25, 0.25]

        targets = [
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
        ]

        for target in targets:
            result = rb._non_convex_risk_budget_objective(x, self.cov_4x4, target)
            # All should be non-negative
            self.assertGreaterEqual(result, 0)


@unittest.skipIf(not CVXPY_AVAILABLE, "cvxpy not installed")
class TestRiskContribution(unittest.TestCase):
    """Test cases for risk contribution calculations (internal function)"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov_4x4 = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

    def _calculate_risk_contribution(self, w, cov):
        """Helper function to test risk contribution calculation"""
        w = np.asmatrix(w)
        cov = np.asmatrix(cov)
        port_variance = (w * cov * w.T)[0, 0]
        port_sigma = np.sqrt(port_variance)
        mrc = cov * w.T
        risk_contr = np.multiply(mrc, w.T) / port_sigma
        return risk_contr

    def test_risk_contributions_sum_to_portfolio_risk(self):
        """Test that risk contributions sum to total portfolio risk"""
        weights = [0.25, 0.25, 0.25, 0.25]
        w = np.asmatrix(weights)
        cov = np.asmatrix(self.cov_4x4)

        # Calculate portfolio risk
        port_variance = (w * cov * w.T)[0, 0]
        port_sigma = np.sqrt(port_variance)

        # Calculate risk contributions
        risk_contribs = self._calculate_risk_contribution(weights, self.cov_4x4)

        # Risk contributions should sum to portfolio risk
        self.assertAlmostEqual(np.sum(risk_contribs), port_sigma, places=10)

    def test_risk_contributions_positive(self):
        """Test that risk contributions are positive for positive weights"""
        weights = [0.25, 0.25, 0.25, 0.25]
        risk_contribs = self._calculate_risk_contribution(weights, self.cov_4x4)

        # All risk contributions should be positive
        self.assertTrue(np.all(risk_contribs > 0))

    def test_risk_contribution_proportions(self):
        """Test risk contribution proportions"""
        weights = [0.4, 0.3, 0.2, 0.1]
        w = np.asmatrix(weights)
        cov = np.asmatrix(self.cov_4x4)

        port_variance = (w * cov * w.T)[0, 0]
        port_sigma = np.sqrt(port_variance)

        risk_contribs = self._calculate_risk_contribution(weights, self.cov_4x4)

        # Risk contribution fractions should sum to 1
        risk_fractions = risk_contribs / port_sigma
        self.assertAlmostEqual(np.sum(risk_fractions), 1.0, places=10)


@unittest.skipIf(not CVXPY_AVAILABLE, "cvxpy not installed")
class TestRiskBudgetingIntegration(unittest.TestCase):
    """Integration tests for risk budgeting optimization"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov_4x4 = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

    def test_convex_solution_properties(self):
        """Test that convex solution has correct properties"""
        x_target = [0.25, 0.25, 0.25, 0.25]
        result = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, x_target)

        # Weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # All weights non-negative
        self.assertTrue(all(w >= 0 for w in result))

        # Objective value should be small
        obj_value = rb._non_convex_risk_budget_objective(result, self.cov_4x4, x_target)
        self.assertLess(obj_value, 1e-3)

    def test_convex_better_than_equal_weight(self):
        """Test that convex solution is better than equal weight"""
        x_target = [0.25, 0.25, 0.25, 0.25]
        x_equal = [0.25, 0.25, 0.25, 0.25]
        x_optimal = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, x_target)

        obj_equal = rb._non_convex_risk_budget_objective(x_equal, self.cov_4x4, x_target)
        obj_optimal = rb._non_convex_risk_budget_objective(x_optimal, self.cov_4x4, x_target)

        # Optimal should be better (lower objective)
        self.assertLessEqual(obj_optimal, obj_equal + 1e-6)

    def test_different_risk_budgets(self):
        """Test various risk budget targets"""
        risk_budgets = [
            [0.25, 0.25, 0.25, 0.25],  # Equal risk parity
            [0.4, 0.3, 0.2, 0.1],      # Decreasing risk
            [0.1, 0.2, 0.3, 0.4],      # Increasing risk
            [0.5, 0.3, 0.15, 0.05],    # Concentrated risk
        ]

        for budget in risk_budgets:
            result = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, budget)

            # All should satisfy basic constraints
            self.assertAlmostEqual(sum(result), 1.0, places=6)
            self.assertTrue(all(w >= 0 for w in result))

    def test_high_variance_asset_gets_lower_weight(self):
        """Test that high variance assets get lower weights in ERP"""
        # Create covariance with one very high variance asset
        cov = np.diag([1.0, 1.0, 10.0, 1.0])
        x_target = [0.25, 0.25, 0.25, 0.25]

        result = rb.solve_convex_risk_budget_obj_func(cov, x_target)

        # Asset 2 (high variance) should have lowest weight
        self.assertEqual(np.argmin(result), 2)

    def test_risk_parity_achieved(self):
        """Test that equal risk parity actually achieves equal risk contributions"""
        x_target = [0.25, 0.25, 0.25, 0.25]
        x_optimal = rb.solve_convex_risk_budget_obj_func(self.cov_4x4, x_target)

        # Calculate actual risk contributions
        w = np.asmatrix(x_optimal)
        cov = np.asmatrix(self.cov_4x4)
        port_variance = (w * cov * w.T)[0, 0]
        port_sigma = np.sqrt(port_variance)
        mrc = cov * w.T
        risk_contrib = np.multiply(mrc, w.T) / port_sigma

        # Convert to fractions
        risk_fractions = np.array(risk_contrib.flat)
        risk_fractions = risk_fractions / port_sigma

        # All risk fractions should be close to 0.25
        for rc in risk_fractions:
            self.assertAlmostEqual(rc, 0.25, places=2)

    def test_2x2_risk_parity(self):
        """Test risk parity with 2x2 matrix"""
        cov_2x2 = np.array([
            [1.0, 0.3],
            [0.3, 4.0]
        ])
        x_target = [0.5, 0.5]

        result = rb.solve_convex_risk_budget_obj_func(cov_2x2, x_target)

        # Weights should sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Lower variance asset should have higher weight
        self.assertGreater(result[0], result[1])


if __name__ == '__main__':
    unittest.main()
