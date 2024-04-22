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
        # Check that returns is a DataFrame
        if not isinstance(price_df, pd.DataFrame):
            raise ValueError("returns must be a DataFrame")

        # Check that risk_free_rate is a float
        if not isinstance(risk_free_rate, float):
            raise ValueError("risk_free_rate must be a float")

        self.price_df = price_df
        self.risk_free_rate = risk_free_rate

    def calculate_mu_S(self):
        mu = expected_returns.mean_historical_return(self.price_df)
        S = risk_models.sample_cov(self.price_df)
        return mu, S

    def maximize_portfolio_sharpe_ratio(self, expected_returns, cov_matrix):

        # Objective function: minimize the negative of the Sharpe Ratio
        def neg_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std_dev
            return -sharpe_ratio

        # Initial guess for portfolio weights (equal distribution)
        initial_weights = np.full(len(expected_returns), 1 / len(expected_returns))

        # Constraints: Sum of weights equals 1
        constraints = LinearConstraint(np.ones(len(expected_returns)), lb=1, ub=1)

        # Assuming no short selling and weights sum to 1
        bounds = Bounds(0, 1)

        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
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

    def maximize_sharpe_ratio_with_sectors(
        self, expected_returns, cov_matrix, risk_free_rate, sector_map, sector_bounds
    ):
        # sector_map = ['Financials', 'Financials', 'Financials', 'Telecommunications', 'Industrials', 'Industrials', 'Real Estate', 'Energy', 'Industrials', 'Real Estate', 'Consumer Discretionary']
        # sector_bounds = {
        #     "Financials": (0.2, 0.5),
        #     "Real Estate": (0.1, 0.2),
        #     "Telecommunications": (0.05, 0.15),
        #     "Consumer Discretionary": (0.1, 0.2),
        #     "Industrials": (0.1, 0.2),
        # }
        # Number of assets
        n_assets = len(expected_returns)

        # Objective function: minimize the negative of the Sharpe Ratio
        def neg_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return -sharpe_ratio  # Minimizing negative Sharpe Ratio maximizes it

        # Initial guess for portfolio weights (equal distribution)
        initial_weights = np.full(n_assets, 1 / n_assets)

        # General constraint: Sum of weights equals 1
        constraints = [LinearConstraint(np.ones(n_assets), lb=1, ub=1)]

        # Sector constraints
        for sector, bounds in sector_bounds.items():
            sector_indices = [
                i for i, asset in enumerate(sector_map) if asset == sector
            ]
            sector_constraint = LinearConstraint(
                np.eye(n_assets)[sector_indices], lb=bounds[0], ub=bounds[1]
            )
            constraints.append(sector_constraint)

        # Bounds for weights, assuming no short selling and weights sum to 1
        bounds = Bounds(0, 1)

        # Solve the optimization problem using the trust-constr method
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            options={"verbose": 1},
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
