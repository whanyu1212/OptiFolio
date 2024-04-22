import pandas as pd


class Backtest:
    def __init__(
        self, holdout_df: pd.DataFrame, weights: dict, initial_fund: float = 1000000.0
    ):
        self.holdout_df = holdout_df
        self.weights = weights
        self.initial_fund = initial_fund

    def calculate_fund_allocation(self):
        # Calculate the fund allocation based on the weights
        starting_price = self.holdout_df.iloc[0]
        fund_list = [weight * self.initial_fund for weight in self.weights.values()]

        # Perform element-wise division
        shares, remainder = divmod(fund_list, starting_price)

        return shares, remainder

    def calculate_daily_capital(self, shares):
        # multiply the shares by the daily price
        daily_prices = self.holdout_df
        daily_capital = daily_prices * shares
        return daily_capital

    def evaluate_portfolio_performance(self, daily_capital):
        # row sum of daily capital
        portfolio_value = daily_capital.sum(axis=1)
        portfolio_daily_return = portfolio_value.pct_change().dropna()
        portfolio_sharpe_ratio = (
            portfolio_daily_return.mean()
        ) / portfolio_daily_return.std()
        return portfolio_sharpe_ratio
