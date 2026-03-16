# Portfolio Construction Test Suite

This directory contains comprehensive unit tests for the portfolio construction modules.

## Test Coverage

### 1. `test_scipy_port_constraints.py`
Tests for constraint functions used in scipy-based optimization:
- **TestTotalWeightConstraint**: Tests for `total_weight_constraint()` function
  - Weights summing to 1, greater than 1, less than 1
  - Edge cases: empty arrays, single weight, large portfolios
  - Numerical precision tests

- **TestLongOnlyConstraint**: Tests for `long_only_constraint()` function
  - All positive, negative, and mixed weights
  - Array type preservation

- **TestConstraintsIntegration**: Integration tests for both constraints working together

### 2. `test_max_diversification.py`
Tests for Maximum Diversification Portfolio (MDP):
- **TestCalcDiversificationRatio**: Tests for `_calc_diversification_ratio()` function
  - Equal and concentrated portfolios
  - Mathematical properties validation
  - Uncorrelated assets

- **TestSolveMDPWeights**: Tests for `solve_mdp_weights()` function
  - Basic MDP solutions with/without bounds
  - Long-only constraint testing
  - Different initial guesses
  - Bounds handling and warnings

- **TestMDPIntegration**: Integration tests verifying MDP produces diversified portfolios

### 3. `test_hrp.py`
Tests for Hierarchical Risk Parity (HRP):
- **TestCovToCorrMatrix**: Tests for `cov_to_corr_matrix()` function
  - Identity and known covariance matrices
  - Symmetry preservation
  - Correlation value bounds

- **TestDistanceCalc**: Tests for `distance_calc()` function
  - Perfect correlation, zero correlation, negative correlation
  - Distance function properties

- **TestGetQuasiDiag**: Tests for `get_quasi_diag()` function
  - Output length and completeness
  - No duplicates, all integers

- **TestGetClusterVar**: Tests for `get_cluster_var()` function
  - Single asset, multi-asset clusters
  - Positive variance validation

- **TestGetRecBipart**: Tests for `get_rec_bipart()` function
  - Weights sum to 1
  - All weights positive
  - Correct indexing

- **TestHRPIntegration**: Full workflow integration tests

### 4. `test_risk_budget.py`
Tests for Risk Budgeting/Parity optimization:
- **TestSolveConvexRiskBudget**: Tests for `solve_convex_risk_budget_obj_func()` function
  - Equal risk parity solutions
  - Unequal risk targets
  - Diagonal covariance matrices

- **TestNonConvexRiskBudgetObjective**: Tests for `_non_convex_risk_budget_objective()` function
  - Objective function properties
  - Optimality verification

- **TestRiskContribution**: Tests for risk contribution calculations
  - Risk contributions sum to portfolio risk
  - Positive contributions
  - Proportion validation

- **TestRiskBudgetingIntegration**: Integration tests verifying risk parity is achieved

## Running the Tests

### Run all tests:
```bash
# From this directory
python run_all_tests.py

# Or using unittest discovery from project root
python -m unittest discover finance/portfolio_construction/tests

# Or from the tests directory
cd finance/portfolio_construction/tests
python -m unittest discover
```

### Run specific test file:
```bash
# Run constraints tests
python -m unittest test_scipy_port_constraints

# Run MDP tests
python -m unittest test_max_diversification

# Run HRP tests
python -m unittest test_hrp

# Run risk budgeting tests
python -m unittest test_risk_budget
```

### Run specific test class:
```bash
python -m unittest test_scipy_port_constraints.TestTotalWeightConstraint
```

### Run specific test method:
```bash
python -m unittest test_scipy_port_constraints.TestTotalWeightConstraint.test_weights_sum_to_one
```

### Run with verbose output:
```bash
python -m unittest discover -v
```

## Test Statistics

- **Total test files**: 4
- **Total test classes**: 17
- **Total test methods**: 100+
- **Code coverage**: Tests all public functions and key edge cases

## Dependencies

The tests require the following packages:
- `numpy` - Numerical operations
- `pandas` - Data structures
- `scipy` - Optimization and clustering
- `cvxpy` - Convex optimization (for risk budgeting tests)
- `unittest` - Python's built-in testing framework

## Test Design Principles

1. **Isolation**: Each test is independent and doesn't rely on others
2. **Comprehensive**: Tests cover normal cases, edge cases, and error conditions
3. **Numerical accuracy**: Uses appropriate tolerance levels for floating-point comparisons
4. **Mathematical validation**: Verifies mathematical properties (e.g., weights sum to 1, correlations in [-1, 1])
5. **Integration testing**: Tests complete workflows to ensure components work together

## Adding New Tests

When adding new tests:
1. Follow the existing naming convention: `test_<module_name>.py`
2. Organize tests into logical test classes
3. Use descriptive test method names that explain what is being tested
4. Include docstrings explaining the test purpose
5. Use `setUp()` methods for common test fixtures
6. Test both success and failure cases
7. Verify numerical properties with appropriate tolerances

## Continuous Integration

These tests can be integrated into CI/CD pipelines:
```yaml
# Example for GitHub Actions
- name: Run tests
  run: |
    cd finance/portfolio_construction/tests
    python run_all_tests.py
```

## Known Issues / Limitations

- Some optimization tests may show minor numerical differences across platforms due to solver implementations
- Tests use tolerance levels (e.g., `places=6`) to account for floating-point precision
- Warnings from optimization solvers are expected in some tests
