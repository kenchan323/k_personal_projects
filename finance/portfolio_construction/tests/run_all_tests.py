#!/usr/bin/env python
"""
Test runner script for all portfolio construction tests.

Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py -v           # Run with verbose output
    python run_all_tests.py <module>     # Run specific test module
"""

import sys
import os
import unittest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def run_all_tests(verbosity=2):
    """Run all test modules"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_specific_test(module_name, verbosity=2):
    """Run a specific test module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] not in ['-v', '--verbose']:
        # Run specific test module
        module = sys.argv[1]
        verbose = '-v' in sys.argv or '--verbose' in sys.argv
        success = run_specific_test(module, verbosity=2 if verbose else 1)
    else:
        # Run all tests
        verbose = '-v' in sys.argv or '--verbose' in sys.argv
        success = run_all_tests(verbosity=2 if verbose else 1)

    sys.exit(0 if success else 1)
