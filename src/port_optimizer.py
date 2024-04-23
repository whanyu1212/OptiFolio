import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from pypfopt import (
    EfficientFrontier,
    risk_models,
    expected_returns,
    objective_functions,
)
from scipy.optimize import minimize, Bounds, LinearConstraint
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


class PortfolioOptimizer:
    def __init__(self, price_df: pd.DataFrame, risk_free_rate: float = 0.0):
        # Check that price_df is a DataFrame
        if not isinstance(price_df, pd.DataFrame):
            raise ValueError("price_df must be a DataFrame")

        # Check that risk_free_rate is a float
        if not isinstance(risk_free_rate, float):
            raise ValueError("risk_free_rate must be a float")

        self.price_df = price_df
        self.risk_free_rate = risk_free_rate

    def calculate_mu_S(self):
        mu = expected_returns.mean_historical_return(self.price_df)
        S = risk_models.sample_cov(self.price_df)
        return mu, S

    def neg_sharpe_ratio(self, weights, expected_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe_ratio  # Minimizing negative Sharpe Ratio maximizes it

    def maximize_sharpe_ratio_by_mc(
        self, expected_returns, cov_matrix, iterations=10000
    ):
        np.random.seed(42)

        min_sharpe = 0

        for i in range(iterations):
            weights = np.random.random(len(self.price_df.columns))
            weights /= np.sum(weights)
            sharpe = self.neg_sharpe_ratio(
                weights, expected_returns, cov_matrix, self.risk_free_rate
            )
            if sharpe < min_sharpe:
                min_sharpe = sharpe
                optimal_w = weights
        return optimal_w

    def maximize_sharpe_ratio_base(self, expected_returns, cov_matrix):
        # Initial guess for portfolio weights (equal distribution)
        initial_weights = np.full(len(expected_returns), 1 / len(expected_returns))

        # Constraints: Sum of weights equals 1
        constraints = LinearConstraint(np.ones(len(expected_returns)), lb=1, ub=1)

        # Assuming no short selling and weights sum to 1
        bounds = Bounds(0, 1)

        result = minimize(
            self.neg_sharpe_ratio,
            initial_weights,
            args=(expected_returns, cov_matrix, self.risk_free_rate),
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            options={"verbose": 0},
        )

        if result.success:
            return result.x
        else:
            print("Optimization failed:", result.message)
            return None

    def create_sector_constraints(self, sector_map, sector_bounds, n_assets):
        constraints = []
        for sector, bounds in sector_bounds.items():
            sector_indices = [
                i for i, asset in enumerate(sector_map) if asset == sector
            ]
            sector_constraint = LinearConstraint(
                np.eye(n_assets)[sector_indices], lb=bounds[0], ub=bounds[1]
            )
            constraints.append(sector_constraint)
        return constraints

    def maximize_sharpe_ratio_with_sectors(
        self, expected_returns, cov_matrix, risk_free_rate, sector_map, sector_bounds
    ):
        n_assets = len(expected_returns)

        initial_weights = np.full(n_assets, 1 / n_assets)

        constraints = [LinearConstraint(np.ones(n_assets), lb=1, ub=1)]
        constraints += self.create_sector_constraints(
            sector_map, sector_bounds, n_assets
        )

        bounds = Bounds(0, 1)

        result = minimize(
            self.neg_sharpe_ratio,
            initial_weights,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            options={"verbose": 0},
        )

        if result.success:
            return result.x
        else:
            print("Optimization failed:", result.message)


if __name__ == "__main__":
    # Load data
    price_df = pd.read_csv("./data/price.csv", index_col="Date", parse_dates=True)

    # Initialize PortfolioOptimizer
    po = PortfolioOptimizer(price_df)

    # Calculate expected returns and covariance matrix
    mu, S = po.calculate_mu_S()

    # Optimize portfolio for maximum Sharpe ratio
    sector_map = [
        "Financials",
        "Financials",
        "Financials",
        "Telecommunications",
        "Industrials",
        "Industrials",
        "Real Estate",
        "Energy",
        "Industrials",
        "Real Estate",
        "Consumer Discretionary",
    ]
    sector_bounds = {
        "Financials": (0.1, 0.6),
        "Real Estate": (0.05, 0.3),
        "Telecommunications": (0.02, 0.2),
        "Consumer Discretionary": (0.05, 0.3),
        "Industrials": (0.05, 0.3),
        "Energy": (0.02, 0.2),
    }
    optimized_weights = po.maximize_sharpe_ratio_with_sectors(
        mu, S, 0.0, sector_map, sector_bounds
    )
    for weight in optimized_weights:
        print(weight)
    print("\n")
    print("Optimized weights without sectors:")
    optimized_weights_without_sectors = po.maximize_portfolio_sharpe_ratio(mu, S)
    for weight in optimized_weights_without_sectors:
        print(weight)
    print(type(optimized_weights_without_sectors))
    print("\n")
    print("Optimized weights by Monte Carlo:")
    optimized_weights_by_mc = po.maximize_portfolio_sharpe_by_mc(mu, S)
    for weight in optimized_weights_by_mc:
        print(weight)
