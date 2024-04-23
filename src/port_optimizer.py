import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from typing import Tuple, List, Dict
from loguru import logger
from pypfopt import (
    risk_models,
    expected_returns,
)
from scipy.optimize import minimize, Bounds, LinearConstraint


class PortfolioOptimizer:
    def __init__(self, price_df: pd.DataFrame, risk_free_rate: float = 0.0):
        """Initialize the PortfolioOptimizer class

        Args:
            price_df (pd.DataFrame): Closing price data for assets
            risk_free_rate (float, optional): Defaults to 0.0.

        Raises:
            ValueError: if price_df is not a DataFrame
            ValueError: if risk_free_rate is not a float
        """
        # Check that price_df is a DataFrame
        if not isinstance(price_df, pd.DataFrame):
            raise ValueError("price_df must be a DataFrame")

        # Check that risk_free_rate is a float
        if not isinstance(risk_free_rate, float):
            raise ValueError("risk_free_rate must be a float")

        self.price_df = price_df
        self.risk_free_rate = risk_free_rate

    def calculate_mu_S(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and covariance matrix of asset returns

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of mean returns and covariance matrix
        """
        mu = expected_returns.mean_historical_return(self.price_df)
        S = risk_models.sample_cov(self.price_df)
        return mu, S

    def neg_sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float,
    ) -> float:
        """Calculate the negative Sharpe ratio for a given set of weights.
        The optimization algorithm will minimize this objective function.

        Args:
            weights (np.ndarray): an array of portfolio weights
            expected_returns (np.ndarray): mu calculated from historical returns
            cov_matrix (np.ndarray): covariance matrix of asset returns
            risk_free_rate (float): risk-free rate

        Returns:
            float: negative Sharpe ratio
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return -sharpe_ratio  # Minimizing negative Sharpe Ratio maximizes it

    def maximize_sharpe_ratio_by_mc(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray, iterations=10000
    ) -> np.ndarray:
        """Maximize the Sharpe ratio using Monte Carlo simulation

        Args:
            expected_returns (np.ndarray): mu calculated from historical returns
            cov_matrix (np.ndarray): covariance matrix of asset returns
            iterations (int, optional): Defaults to 10000.

        Returns:
            np.ndarray: optimal portfolio weights
        """
        np.random.seed(42)

        min_sharpe = 0

        for _ in range(iterations):
            weights = np.random.random(len(self.price_df.columns))
            weights /= np.sum(weights)
            sharpe = self.neg_sharpe_ratio(
                weights, expected_returns, cov_matrix, self.risk_free_rate
            )
            if sharpe < min_sharpe:
                min_sharpe = sharpe
                optimal_w = weights
        return optimal_w

    def maximize_sharpe_ratio_base(
        self, expected_returns: np.ndarray, cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Baseline optimization function to maximize the Sharpe ratio
        using Scipy's minimize function

        Args:
            expected_returns (np.ndarray): mu calculated from historical returns
            cov_matrix (np.ndarray): covariance matrix of asset returns

        Returns:
            np.ndarray: optimal portfolio weights
        """
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
            method="trust-constr",  # there are also other methods available
            bounds=bounds,
            constraints=constraints,
            options={"verbose": 0},
        )

        if result.success:
            logger.success("Optimization converged successfully")
            return result.x
        else:
            logger.error("Optimization failed: {}", result.message)
            return None

    def create_sector_constraints(
        self, sector_map: List[str], sector_bounds: Dict, n_assets: int
    ) -> List[LinearConstraint]:
        """Create sector constraints for the portfolio optimization

        Args:
            sector_map (List[str]): A list of sectors for each asset
            sector_bounds (Dict): A dictionary of sector bounds
            n_assets (int): Number of assets

        Returns:
            List[LinearConstraint]: A list of sector constraints
        """
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
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float,
        sector_map: List[str],
        sector_bounds: Dict,
    ) -> np.ndarray:
        """Maximize the Sharpe ratio with sector constraints

        Args:
            expected_returns (np.ndarray): mu calculated from historical returns
            cov_matrix (np.ndarray): covariance matrix of asset returns
            risk_free_rate (float): risk-free rate
            sector_map (List[str]): A list of sectors for each asset
            sector_bounds (Dict): A dictionary of sector bounds

        Returns:
            np.ndarray: optimal portfolio weights
        """
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
            logger.success("Optimization converged successfully")
            return result.x
        else:
            logger.error("Optimization failed: {}", result.message)
            return None


# sample usage

# if __name__ == "__main__":
#     # Load data
#     price_df = pd.read_csv("./data/price.csv", index_col="Date", parse_dates=True)

#     # Initialize PortfolioOptimizer
#     po = PortfolioOptimizer(price_df)

#     # Calculate expected returns and covariance matrix
#     mu, S = po.calculate_mu_S()

#     # Optimize portfolio for maximum Sharpe ratio
#     sector_map = [
#         "Financials",
#         "Financials",
#         "Financials",
#         "Telecommunications",
#         "Industrials",
#         "Industrials",
#         "Real Estate",
#         "Energy",
#         "Industrials",
#         "Real Estate",
#         "Consumer Discretionary",
#     ]
#     sector_bounds = {
#         "Financials": (0.1, 0.6),
#         "Real Estate": (0.05, 0.3),
#         "Telecommunications": (0.02, 0.2),
#         "Consumer Discretionary": (0.05, 0.3),
#         "Industrials": (0.05, 0.3),
#         "Energy": (0.02, 0.2),
#     }
#     optimized_weights = po.maximize_sharpe_ratio_with_sectors(
#         mu, S, 0.0, sector_map, sector_bounds
#     )
#     for weight in optimized_weights:
#         print(weight)
#     print("\n")
#     print("Optimized weights without sectors:")
#     optimized_weights_without_sectors = po.maximize_portfolio_sharpe_ratio(mu, S)
#     for weight in optimized_weights_without_sectors:
#         print(weight)
#     print(type(optimized_weights_without_sectors))
#     print("\n")
#     print("Optimized weights by Monte Carlo:")
#     optimized_weights_by_mc = po.maximize_portfolio_sharpe_by_mc(mu, S)
#     for weight in optimized_weights_by_mc:
#         print(weight)
