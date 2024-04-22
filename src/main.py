import pandas as pd
import yfinance as yf
from typing import Tuple
from src.data_loader import DataLoader
from src.port_optimizer import PortfolioOptimizer
from src.backtest import Backtest
from src.utils import parse_yaml_cfg


def get_cfg(cfg_path: str) -> dict:
    """Get config from yaml file

    Args:
        cfg_path (str): path to yaml file

    Returns:
        dict: config in dictionary format
    """
    return parse_yaml_cfg(cfg_path)


def load_data(cfg: dict) -> pd.DataFrame:
    """Load data from the data loader

    Args:
        cfg (dict): config dictionary

    Returns:
        pd.DataFrame: data
    """
    dl = DataLoader(
        tickers=cfg["tickers"],
        benchmark_ticker=cfg["benchmark_ticker"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
    )
    train, holdout = dl.train_test_split(cfg["train_test_cut_off_date"])
    return train, holdout


def optimize_portfolio(df: pd.DataFrame) -> dict:
    """Optimize the portfolio for a target return

    Args:
        df (pd.DataFrame): A DataFrame of returns data

    Returns:
        dict: A dictionary containing the optimized portfolio weights
    """
    po = PortfolioOptimizer(df)
    mu, S = po.calculate_mu_S()
    optimized_weights = po.maximize_portfolio_sharpe_ratio(mu, S)
    keys = list(mu.index)
    values = optimized_weights

    weights_dict = dict(zip(keys, values))
    return weights_dict


def backtest_hold_out(holdout: pd.DataFrame, optimized_weights: dict):
    """Backtest the optimized portfolio

    Args:
        holdout (pd.DataFrame): Holdout data
        optimized_weights (dict): Optimized portfolio weights
    """
    backtest = Backtest(holdout, optimized_weights)
    shares, remainder = backtest.calculate_fund_allocation()
    print(shares, remainder)
    daily_capital = backtest.calculate_daily_capital(shares)
    sharpe_ratio = backtest.evaluate_portfolio_performance(daily_capital)
    print(sharpe_ratio)


def main():
    cfg = get_cfg("./cfg/parameters.yaml")
    train, holdout = load_data(cfg)
    optimized_weights = optimize_portfolio(
        train.drop(columns=cfg["benchmark_ticker"], axis=1)
    )
    print(optimized_weights)
    backtest_hold_out(
        holdout.drop(columns=cfg["benchmark_ticker"], axis=1), optimized_weights
    )


if __name__ == "__main__":
    main()
