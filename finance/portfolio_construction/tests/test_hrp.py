import unittest
import numpy as np
import pandas as pd
import sys
import os
from scipy.cluster.hierarchy import linkage

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from finance.portfolio_construction import script_hrp as hrp


class TestCovToCorrMatrix(unittest.TestCase):
    """Test cases for cov_to_corr_matrix function"""

    def test_identity_covariance(self):
        """Test with identity matrix (uncorrelated, unit variance)"""
        cov = np.eye(3)
        result = hrp.cov_to_corr_matrix(cov)

        # Should return identity matrix
        np.testing.assert_array_almost_equal(result, np.eye(3))

    def test_known_covariance_matrix(self):
        """Test with known covariance matrix"""
        cov = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.6],
            [0.3, 0.6, 3.0]
        ])

        result = hrp.cov_to_corr_matrix(cov)

        # Diagonal should be all 1s
        np.testing.assert_array_almost_equal(np.diag(result), np.ones(3))

        # Check specific correlations
        # corr[0,1] = cov[0,1] / sqrt(cov[0,0] * cov[1,1]) = 0.5 / sqrt(1*2) = 0.5/1.414 ≈ 0.3536
        expected_corr_01 = 0.5 / np.sqrt(1.0 * 2.0)
        self.assertAlmostEqual(result[0, 1], expected_corr_01, places=6)

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(result, result.T)

    def test_2x2_matrix(self):
        """Test with 2x2 covariance matrix"""
        cov = np.array([
            [1.23, 0.375],
            [0.375, 1.22]
        ])

        result = hrp.cov_to_corr_matrix(cov)

        # Diagonal should be 1s
        self.assertAlmostEqual(result[0, 0], 1.0)
        self.assertAlmostEqual(result[1, 1], 1.0)

        # Off-diagonal should match formula
        expected = 0.375 / np.sqrt(1.23 * 1.22)
        self.assertAlmostEqual(result[0, 1], expected, places=6)
        self.assertAlmostEqual(result[1, 0], expected, places=6)

    def test_4x4_matrix(self):
        """Test with standard 4x4 matrix from script"""
        cov = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

        result = hrp.cov_to_corr_matrix(cov)

        # Diagonal should be 1s
        np.testing.assert_array_almost_equal(np.diag(result), np.ones(4))

        # Should be symmetric
        np.testing.assert_array_almost_equal(result, result.T)

        # Correlation values should be between -1 and 1
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_symmetry_preservation(self):
        """Test that correlation matrix is symmetric"""
        cov = np.random.rand(5, 5)
        cov = (cov + cov.T) / 2  # Make symmetric
        np.fill_diagonal(cov, np.abs(np.diag(cov)) + 1)  # Ensure positive diagonal

        result = hrp.cov_to_corr_matrix(cov)

        # Should be symmetric
        np.testing.assert_array_almost_equal(result, result.T)


class TestDistanceCalc(unittest.TestCase):
    """Test cases for distance_calc function"""

    def test_perfect_correlation(self):
        """Test with perfect positive correlation"""
        result = hrp.distance_calc(1.0)
        self.assertEqual(result, 0.0)

    def test_zero_correlation(self):
        """Test with zero correlation"""
        result = hrp.distance_calc(0.0)
        expected = np.sqrt(0.5)
        self.assertAlmostEqual(result, expected, places=10)

    def test_negative_correlation(self):
        """Test with negative correlation"""
        result = hrp.distance_calc(-0.5)
        expected = np.sqrt(0.5 * (1 - (-0.5)))
        self.assertAlmostEqual(result, expected, places=10)

    def test_perfect_negative_correlation(self):
        """Test with perfect negative correlation"""
        result = hrp.distance_calc(-1.0)
        expected = np.sqrt(0.5 * 2.0)
        self.assertAlmostEqual(result, expected, places=10)

    def test_positive_correlation(self):
        """Test with positive correlation"""
        result = hrp.distance_calc(0.7)
        expected = np.sqrt(0.5 * (1 - 0.7))
        self.assertAlmostEqual(result, expected, places=10)

    def test_distance_properties(self):
        """Test that distance function has correct properties"""
        # Distance should be 0 for correlation of 1
        self.assertEqual(hrp.distance_calc(1.0), 0.0)

        # Distance should increase as correlation decreases
        d1 = hrp.distance_calc(0.9)
        d2 = hrp.distance_calc(0.5)
        d3 = hrp.distance_calc(0.0)
        self.assertLess(d1, d2)
        self.assertLess(d2, d3)

        # All distances should be non-negative
        test_corrs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for corr in test_corrs:
            self.assertGreaterEqual(hrp.distance_calc(corr), 0.0)


class TestGetQuasiDiag(unittest.TestCase):
    """Test cases for get_quasi_diag function"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a simple correlation matrix
        corr = np.array([
            [1.0, 0.8, 0.3, 0.2],
            [0.8, 1.0, 0.25, 0.15],
            [0.3, 0.25, 1.0, 0.9],
            [0.2, 0.15, 0.9, 1.0]
        ])

        # Convert to distance matrix
        dist_df = pd.DataFrame(corr).map(lambda x: hrp.distance_calc(x))

        # Create linkage matrix
        self.link = linkage(dist_df, 'single', optimal_ordering=True)

    def test_quasi_diag_output_length(self):
        """Test that output has correct length"""
        result = hrp.get_quasi_diag(self.link)
        self.assertEqual(len(result), 4)

    def test_quasi_diag_contains_all_indices(self):
        """Test that output contains all original indices"""
        result = hrp.get_quasi_diag(self.link)
        result_set = set(result)

        # Should contain indices 0, 1, 2, 3
        expected_set = {0, 1, 2, 3}
        self.assertEqual(result_set, expected_set)

    def test_quasi_diag_no_duplicates(self):
        """Test that output has no duplicate indices"""
        result = hrp.get_quasi_diag(self.link)
        self.assertEqual(len(result), len(set(result)))

    def test_quasi_diag_all_integers(self):
        """Test that all returned values are integers"""
        result = hrp.get_quasi_diag(self.link)
        for item in result:
            self.assertIsInstance(int(item), int)


class TestGetClusterVar(unittest.TestCase):
    """Test cases for get_cluster_var function"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov = pd.DataFrame([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

    def test_single_asset_cluster(self):
        """Test cluster variance for single asset"""
        result = hrp.get_cluster_var(self.cov, [0])
        # For single asset, cluster variance should equal asset variance
        self.assertAlmostEqual(result, self.cov.iloc[0, 0], places=10)

    def test_two_asset_cluster(self):
        """Test cluster variance for two assets"""
        result = hrp.get_cluster_var(self.cov, [0, 1])
        # Should be positive
        self.assertGreater(result, 0)

    def test_all_assets_cluster(self):
        """Test cluster variance for all assets"""
        result = hrp.get_cluster_var(self.cov, [0, 1, 2, 3])
        # Should be positive
        self.assertGreater(result, 0)

    def test_different_clusters(self):
        """Test different cluster combinations"""
        clusters = [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 2],
        ]

        for cluster in clusters:
            result = hrp.get_cluster_var(self.cov, cluster)
            # All should be positive
            self.assertGreater(result, 0)

    def test_cluster_variance_positive(self):
        """Test that cluster variance is always positive"""
        # Test various cluster combinations
        test_clusters = [
            [0],
            [1],
            [2],
            [3],
            [0, 1],
            [2, 3],
            [0, 1, 2],
            [1, 2, 3],
            [0, 1, 2, 3]
        ]

        for cluster in test_clusters:
            result = hrp.get_cluster_var(self.cov, cluster)
            self.assertGreater(result, 0)


class TestGetRecBipart(unittest.TestCase):
    """Test cases for get_rec_bipart function"""

    def setUp(self):
        """Set up test fixtures"""
        self.cov = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

        corr_mat = hrp.cov_to_corr_matrix(self.cov)
        df_dist = pd.DataFrame(corr_mat).map(lambda x: hrp.distance_calc(x))
        link = linkage(df_dist, 'single', optimal_ordering=True)
        self.sorted_index = hrp.get_quasi_diag(link)
        self.cov_df = pd.DataFrame(data=self.cov)

    def test_weights_sum_to_one(self):
        """Test that HRP weights sum to 1"""
        weights = hrp.get_rec_bipart(self.cov_df, self.sorted_index)
        self.assertAlmostEqual(weights.sum(), 1.0, places=10)

    def test_all_weights_positive(self):
        """Test that all HRP weights are positive"""
        weights = hrp.get_rec_bipart(self.cov_df, self.sorted_index)
        self.assertTrue(all(weights >= 0))

    def test_correct_number_of_weights(self):
        """Test that HRP returns correct number of weights"""
        weights = hrp.get_rec_bipart(self.cov_df, self.sorted_index)
        self.assertEqual(len(weights), 4)

    def test_weights_index_matches_sorted_index(self):
        """Test that weights are indexed correctly"""
        weights = hrp.get_rec_bipart(self.cov_df, self.sorted_index)
        # Weights index should match sorted_index
        self.assertEqual(list(weights.index), self.sorted_index)

    def test_hrp_different_from_equal_weight(self):
        """Test that HRP produces different allocation than equal weight"""
        weights = hrp.get_rec_bipart(self.cov_df, self.sorted_index)
        equal_weights = [0.25, 0.25, 0.25, 0.25]

        # At least one weight should differ significantly from 0.25
        differences = [abs(w - 0.25) for w in weights.values]
        self.assertTrue(any(d > 0.01 for d in differences))


class TestHRPIntegration(unittest.TestCase):
    """Integration tests for HRP workflow"""

    def test_full_hrp_workflow(self):
        """Test complete HRP workflow from covariance to weights"""
        # Start with covariance matrix
        cov = np.array([
            [1.23, 0.375, 0.7, 0.3],
            [0.375, 1.22, 0.72, 0.135],
            [0.7, 0.72, 3.21, -0.32],
            [0.3, 0.135, -0.32, 0.52]
        ])

        # Step 1: Convert to correlation
        corr_mat = hrp.cov_to_corr_matrix(cov)

        # Step 2: Convert to distance matrix
        df_dist = pd.DataFrame(corr_mat).map(lambda x: hrp.distance_calc(x))

        # Step 3: Perform hierarchical clustering
        link = linkage(df_dist, 'single', optimal_ordering=True)

        # Step 4: Get quasi-diagonal ordering
        sorted_index = hrp.get_quasi_diag(link)

        # Step 5: Get HRP weights
        weights = hrp.get_rec_bipart(pd.DataFrame(data=cov), sorted_index)

        # Verify final weights
        self.assertAlmostEqual(weights.sum(), 1.0, places=10)
        self.assertTrue(all(weights >= 0))
        self.assertEqual(len(weights), 4)

    def test_hrp_2x2_matrix(self):
        """Test HRP with simple 2x2 matrix"""
        cov = np.array([
            [1.0, 0.5],
            [0.5, 2.0]
        ])

        corr_mat = hrp.cov_to_corr_matrix(cov)
        df_dist = pd.DataFrame(corr_mat).map(lambda x: hrp.distance_calc(x))
        link = linkage(df_dist, 'single', optimal_ordering=True)
        sorted_index = hrp.get_quasi_diag(link)
        weights = hrp.get_rec_bipart(pd.DataFrame(data=cov), sorted_index)

        # Check basic properties
        self.assertAlmostEqual(weights.sum(), 1.0, places=10)
        self.assertTrue(all(weights >= 0))
        self.assertEqual(len(weights), 2)

    def test_hrp_with_high_correlation_cluster(self):
        """Test HRP with assets that have high correlation"""
        # Create matrix where assets 0,1 are highly correlated and 2,3 are highly correlated
        cov = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 2.0, 1.8],
            [0.1, 0.1, 1.8, 2.0]
        ])

        corr_mat = hrp.cov_to_corr_matrix(cov)
        df_dist = pd.DataFrame(corr_mat).map(lambda x: hrp.distance_calc(x))
        link = linkage(df_dist, 'single', optimal_ordering=True)
        sorted_index = hrp.get_quasi_diag(link)
        weights = hrp.get_rec_bipart(pd.DataFrame(data=cov), sorted_index)

        # Weights should favor lower variance assets
        # Assets 0,1 have lower variance than 2,3, so should have higher combined weight
        weight_low_var = weights[0] + weights[1]
        weight_high_var = weights[2] + weights[3]

        self.assertGreater(weight_low_var, weight_high_var)


if __name__ == '__main__':
    unittest.main()
