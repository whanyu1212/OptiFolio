import pandas as pd
import numpy as np
from loguru import logger
from typing import Tuple, List, Dict
from src.data_loader import DataLoader
from src.port_optimizer import PortfolioOptimizer
from src.backtest import Backtest
from src.utils import parse_yaml_cfg


def load_data(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data using the DataLoader
    class initialized

    Args:
        cfg (dict): config dictionary

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train and holdout data
    """
    dl = DataLoader(
        tickers=cfg["tickers"],
        benchmark_ticker=cfg["benchmark_ticker"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
    )
    train, holdout = dl.train_test_split(cfg["train_test_cut_off_date"])
    return train, holdout


def calculate_portfolio_parameters(
    df: pd.DataFrame,
) -> Tuple[PortfolioOptimizer, pd.Series, np.ndarray]:
    """Calculate the mean and covariance matrix of asset returns

    Args:
        df (pd.DataFrame): input DataFrame

    Returns:
        Tuple[PortfolioOptimizer, pd.Series, np.ndarray]: PortfolioOptimizer object,
        mean and covariance matrix
    """
    po = PortfolioOptimizer(df)
    mu, S = po.calculate_mu_S()
    return po, mu, S


def optimize_portfolio_base(
    df: pd.DataFrame,
) -> dict:
    """Optimize the portfolio for a target return

    Args:
        df (pd.DataFrame): A DataFrame of returns data

    Returns:
        dict: A dictionary containing the optimized portfolio weights
    """
    po, mu, S = calculate_portfolio_parameters(df)
    optimized_weights = po.maximize_sharpe_ratio_base(mu, S)
    weights_dict = {key: value for key, value in zip(mu.index, optimized_weights)}
    return weights_dict


def optimize_portfolio_by_mc(
    df: pd.DataFrame,
) -> dict:
    """Optimize the portfolio for a target return

    Args:
        df (pd.DataFrame): A DataFrame of returns data

    Returns:
        dict: A dictionary containing the optimized portfolio weights
    """
    po, mu, S = calculate_portfolio_parameters(df)
    optimized_weights = po.maximize_sharpe_ratio_by_mc(mu, S)
    weights_dict = {key: value for key, value in zip(mu.index, optimized_weights)}
    return weights_dict


def optimize_portfolio_w_sector(
    df: pd.DataFrame,
    sector_map: List[str],
    sector_bounds: Dict[str, Tuple[float, float]],
) -> dict:
    """Optimize the portfolio for a target return

    Args:
        df (pd.DataFrame): A DataFrame of returns data
        sector_map (List[str]): A list mapping sectors to assets
        sector_bounds (Dict[str, Tuple[float, float]]): A dictionary mapping sectors to their bounds

    Returns:
        dict: A dictionary containing the optimized portfolio weights
    """
    po, mu, S = calculate_portfolio_parameters(df)
    optimized_weights_with_sectors = po.maximize_sharpe_ratio_with_sectors(
        mu, S, 0.0, sector_map, sector_bounds
    )
    weights_dict = {
        key: value for key, value in zip(mu.index, optimized_weights_with_sectors)
    }
    return weights_dict


def backtest_hold_out_relative_return(holdout: pd.DataFrame, optimized_weights: dict):
    """Backtest the optimized portfolio

    Args:
        holdout (pd.DataFrame): Holdout data
        optimized_weights (dict): Optimized portfolio weights
    """
    backtest = Backtest(holdout, optimized_weights)
    sharpe_ratio = backtest.evaluate_portfolio_performance_on_relative_returns()
    benchmark_sharpe = backtest.benchmark_buy_and_hold_relative_return()
    return sharpe_ratio, benchmark_sharpe


def backtest_hold_out_dollar_amount(holdout: pd.DataFrame, optimized_weights: dict):
    """Backtest the optimized portfolio

    Args:
        holdout (pd.DataFrame): Holdout data
        optimized_weights (dict): Optimized portfolio weights
    """
    backtest = Backtest(holdout, optimized_weights)
    shares, remainder = backtest.calculate_fund_allocation()
    daily_capital = backtest.calculate_daily_capital(shares)
    sharpe_ratio = backtest.evaluate_portfolio_performance_dollar_amount(daily_capital)
    benchmark_sharpe = backtest.benchmark_buy_and_hold_dollar_amount()
    return sharpe_ratio, benchmark_sharpe


def main():
    logger.info("Starting the analysis")
    cfg = parse_yaml_cfg("./cfg/parameters.yaml")
    sector_map = cfg["sector_map"]

    # There are more sector bounds variation in the config file,
    # but for the sake of simplicity, we will only use the first one
    # for automation purposes
    sector_bounds = {key: tuple(value) for key, value in cfg["sector_bounds_1"].items()}

    # divide the data into train and holdout
    train, holdout = load_data(cfg)

    # optimize the portfolio using the base method
    optimized_weights_base = optimize_portfolio_base(
        train.drop(columns=cfg["benchmark_ticker"], axis=1)
    )

    print(f"Optimized weights without sectors:\n {optimized_weights_base}")
    print("\n")

    # optimize the portfolio by imposing sector constraints
    optimized_weights_w_sector = optimize_portfolio_w_sector(
        train.drop(columns=cfg["benchmark_ticker"], axis=1),
        sector_map,
        sector_bounds,
    )
    print(f"Optimized weights with sectors:\n {optimized_weights_w_sector}")
    print("\n")

    sharpe_base, benchmark_sharpe = backtest_hold_out_relative_return(
        holdout, optimized_weights_base
    )
    print(
        f"Sharpe Ratio without sector constraints: {sharpe_base} vs Benchmark Sharpe Ratio: {benchmark_sharpe}"
    )

    sharpe_w_sectors, benchmark_sharpe = backtest_hold_out_dollar_amount(
        holdout, optimized_weights_w_sector
    )
    print(
        f"Sharpe Ratio by sector: {sharpe_w_sectors} vs Benchmark Sharpe Ratio: {benchmark_sharpe}"
    )

    logger.success("Analysis completed")


if __name__ == "__main__":
    main()
