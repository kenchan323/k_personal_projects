import unittest
import numpy as np
import sys
import os
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from finance.portfolio_construction import script_max_diversification as mdp


class TestCalcDiversificationRatio(unittest.TestCase):
    """Test cases for _calc_diversification_ratio function"""

    def setUp(self):
        """Set up test fixtures"""
        # Simple 2x2 covariance matrix
        self.cov_2x2 = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])

        # Standard 4x4 covariance matrix from the script
        self.cov_4x4 = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

    def test_equal_weights_2x2(self):
        """Test diversification ratio with equal weights on 2x2 matrix"""
        weights = [0.5, 0.5]
        result = mdp._calc_diversification_ratio(weights, self.cov_2x2)

        # Result should be negative (for minimization)
        self.assertLess(result, 0)

    def test_equal_weights_4x4(self):
        """Test diversification ratio with equal weights on 4x4 matrix"""
        weights = [0.25, 0.25, 0.25, 0.25]
        result = mdp._calc_diversification_ratio(weights, self.cov_4x4)

        # Result should be negative (for minimization)
        self.assertLess(result, 0)

    def test_concentrated_portfolio(self):
        """Test with all weight on one asset"""
        weights = [1.0, 0.0, 0.0, 0.0]
        result = mdp._calc_diversification_ratio(weights, self.cov_4x4)

        # With all weight on one asset, DR = -weighted_vol/port_vol
        # Calculate expected value
        w = np.asmatrix(weights)
        cov = np.asmatrix(self.cov_4x4)
        vol = np.diagonal(cov)
        sum_weighted_vol = np.sum(np.multiply(w, vol))
        port_vol = np.sqrt((w * cov * w.T)[0, 0])
        expected_dr = -sum_weighted_vol / port_vol

        self.assertAlmostEqual(result, expected_dr, places=10)

    def test_diversification_ratio_properties(self):
        """Test that diversification ratio has expected mathematical properties"""
        weights = [0.25, 0.25, 0.25, 0.25]

        # Calculate components
        w = np.asmatrix(weights)
        cov = np.asmatrix(self.cov_4x4)
        vol = np.diagonal(cov)

        sum_weighted_vol = np.sum(np.multiply(w, vol))
        port_vol = np.sqrt((w * cov * w.T)[0, 0])

        expected_dr = -sum_weighted_vol / port_vol

        result = mdp._calc_diversification_ratio(weights, self.cov_4x4)

        self.assertAlmostEqual(result, expected_dr, places=10)

    def test_different_weight_combinations(self):
        """Test various weight combinations"""
        test_weights = [
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4],
            [0.7, 0.1, 0.1, 0.1],
        ]

        for weights in test_weights:
            result = mdp._calc_diversification_ratio(weights, self.cov_4x4)
            # All results should be negative
            self.assertLess(result, 0)

    def test_uncorrelated_assets(self):
        """Test with uncorrelated assets (diagonal covariance matrix)"""
        cov_diag = np.diag([1.0, 2.0, 3.0, 4.0])
        weights = [0.25, 0.25, 0.25, 0.25]

        result = mdp._calc_diversification_ratio(weights, cov_diag)

        # Should be negative
        self.assertLess(result, 0)


class TestSolveMDPWeights(unittest.TestCase):
    """Test cases for solve_mdp_weights function"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov_4x4 = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

        self.w_0 = [0.25, 0.25, 0.25, 0.25]

    def test_basic_mdp_solution(self):
        """Test basic MDP solution without bounds"""
        result = mdp.solve_mdp_weights(self.w_0, self.cov_4x4, long_only=True)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check all weights are non-negative (long only)
        self.assertTrue(all(w >= -1e-6 for w in result))

        # Check correct number of weights
        self.assertEqual(len(result), 4)

    def test_mdp_with_bounds(self):
        """Test MDP solution with position limits"""
        bounds = [(0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5)]
        result = mdp.solve_mdp_weights(self.w_0, self.cov_4x4, bnd=bounds, long_only=True)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check all weights respect bounds
        for w in result:
            self.assertGreaterEqual(w, -1e-6)
            self.assertLessEqual(w, 0.5 + 1e-6)

    def test_mdp_long_only_false(self):
        """Test MDP without long-only constraint"""
        result = mdp.solve_mdp_weights(self.w_0, self.cov_4x4, long_only=False)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check correct number of weights
        self.assertEqual(len(result), 4)

    def test_mdp_tight_bounds(self):
        """Test MDP with tight bounds forcing specific allocation"""
        bounds = [(0.2, 0.3), (0.2, 0.3), (0.2, 0.3), (0.2, 0.3)]
        result = mdp.solve_mdp_weights(self.w_0, self.cov_4x4, bnd=bounds, long_only=True)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check all weights respect bounds
        for w in result:
            self.assertGreaterEqual(w, 0.2 - 1e-6)
            self.assertLessEqual(w, 0.3 + 1e-6)

    def test_mdp_different_initial_guess(self):
        """Test MDP with different initial guesses"""
        initial_guesses = [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ]

        results = []
        for w_0 in initial_guesses:
            result = mdp.solve_mdp_weights(w_0, self.cov_4x4, long_only=True)
            results.append(result)

            # Each should sum to 1
            self.assertAlmostEqual(sum(result), 1.0, places=6)

        # All solutions should be similar (optimization should converge to same solution)
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i], decimal=3)

    def test_mdp_2x2_matrix(self):
        """Test MDP with 2x2 covariance matrix"""
        cov_2x2 = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        w_0 = [0.5, 0.5]

        result = mdp.solve_mdp_weights(w_0, cov_2x2, long_only=True)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # Check non-negative weights
        self.assertTrue(all(w >= -1e-6 for w in result))

    def test_mdp_bounds_negative_warning(self):
        """Test that negative bounds are adjusted when long_only=True"""
        bounds = [(-0.1, 0.5), (0, 0.5), (0, 0.5), (0, 0.5)]

        # Should adjust negative bounds and still produce valid solution
        result = mdp.solve_mdp_weights(self.w_0, self.cov_4x4, bnd=bounds, long_only=True)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # All weights should be non-negative (bounds were adjusted)
        self.assertTrue(all(w >= -1e-6 for w in result))

    def test_mdp_diagonal_covariance(self):
        """Test MDP with diagonal (uncorrelated) covariance matrix"""
        cov_diag = np.diag([1.0, 2.0, 3.0, 4.0])
        w_0 = [0.25, 0.25, 0.25, 0.25]

        result = mdp.solve_mdp_weights(w_0, cov_diag, long_only=True)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result), 1.0, places=6)

        # For uncorrelated assets, optimal should favor lower variance assets
        # Weight should decrease as variance increases
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i] + 1e-6, result[i + 1])


class TestMDPIntegration(unittest.TestCase):
    """Integration tests for MDP optimization"""

    def test_mdp_produces_diversified_portfolio(self):
        """Test that MDP produces more diversified portfolio than equal weight"""
        cov = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])
        w_0 = [0.25, 0.25, 0.25, 0.25]

        # Get MDP solution
        w_mdp = mdp.solve_mdp_weights(w_0, cov, long_only=True)

        # Calculate diversification ratios
        dr_mdp = -mdp._calc_diversification_ratio(w_mdp, cov)
        dr_equal = -mdp._calc_diversification_ratio(w_0, cov)

        # MDP should have higher diversification ratio
        self.assertGreater(dr_mdp, dr_equal - 1e-6)


if __name__ == '__main__':
    unittest.main()
