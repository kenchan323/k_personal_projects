import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from finance.portfolio_construction import scipy_port_constraints as pc


class TestTotalWeightConstraint(unittest.TestCase):
    """Test cases for total_weight_constraint function"""

    def test_weights_sum_to_one(self):
        """Test that weights summing to 1 return 0"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_weights_sum_to_one_unequal(self):
        """Test with unequal weights that sum to 1"""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_weights_sum_greater_than_one(self):
        """Test that weights summing to >1 return positive value"""
        weights = np.array([0.5, 0.5, 0.5, 0.5])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_weights_sum_less_than_one(self):
        """Test that weights summing to <1 return negative value"""
        weights = np.array([0.1, 0.1, 0.1, 0.1])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, -0.6, places=10)

    def test_empty_array(self):
        """Test with empty array"""
        weights = np.array([])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, -1.0, places=10)

    def test_single_weight(self):
        """Test with single weight"""
        weights = np.array([1.0])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_large_portfolio(self):
        """Test with large number of assets"""
        n = 100
        weights = np.ones(n) / n
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_numerical_precision(self):
        """Test numerical precision with weights that almost sum to 1"""
        weights = np.array([0.33333333, 0.33333333, 0.33333334])
        result = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(result, 0.0, places=7)


class TestLongOnlyConstraint(unittest.TestCase):
    """Test cases for long_only_constraint function"""

    def test_all_positive_weights(self):
        """Test that all positive weights are returned as is"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)

    def test_all_zero_weights(self):
        """Test with all zero weights"""
        weights = np.array([0.0, 0.0, 0.0, 0.0])
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)

    def test_mixed_weights(self):
        """Test with mixed positive and negative weights"""
        weights = np.array([0.5, -0.1, 0.3, 0.3])
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)

    def test_all_negative_weights(self):
        """Test with all negative weights"""
        weights = np.array([-0.25, -0.25, -0.25, -0.25])
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)

    def test_single_weight(self):
        """Test with single weight"""
        weights = np.array([0.5])
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)

    def test_large_portfolio(self):
        """Test with large number of assets"""
        n = 100
        weights = np.random.rand(n)
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)

    def test_preserves_array_type(self):
        """Test that function preserves numpy array type"""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        result = pc.long_only_constraint(weights)
        self.assertIsInstance(result, np.ndarray)

    def test_very_small_positive_weights(self):
        """Test with very small positive weights"""
        weights = np.array([1e-10, 1e-10, 1e-10, 1.0 - 3e-10])
        result = pc.long_only_constraint(weights)
        np.testing.assert_array_equal(result, weights)


class TestConstraintsIntegration(unittest.TestCase):
    """Integration tests for constraint functions working together"""

    def test_valid_portfolio(self):
        """Test a valid portfolio satisfies both constraints"""
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Should satisfy total weight constraint
        total_constraint = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(total_constraint, 0.0, places=10)

        # Should satisfy long only constraint (all weights >= 0)
        long_only = pc.long_only_constraint(weights)
        self.assertTrue(np.all(long_only >= 0))

    def test_invalid_portfolio_negative_weights(self):
        """Test portfolio with negative weights"""
        weights = np.array([0.5, -0.2, 0.4, 0.3])

        # Still sums to 1
        total_constraint = pc.total_weight_constraint(weights)
        self.assertAlmostEqual(total_constraint, 0.0, places=10)

        # But violates long only
        long_only = pc.long_only_constraint(weights)
        self.assertFalse(np.all(long_only >= 0))

    def test_invalid_portfolio_wrong_sum(self):
        """Test portfolio with wrong sum"""
        weights = np.array([0.3, 0.3, 0.3, 0.3])

        # Doesn't sum to 1
        total_constraint = pc.total_weight_constraint(weights)
        self.assertNotAlmostEqual(total_constraint, 0.0, places=10)

        # But satisfies long only
        long_only = pc.long_only_constraint(weights)
        self.assertTrue(np.all(long_only >= 0))


if __name__ == '__main__':
    unittest.main()
