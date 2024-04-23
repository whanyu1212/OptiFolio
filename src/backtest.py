import pandas as pd
import numpy as np


class Backtest:
    def __init__(
        self, holdout_df: pd.DataFrame, weights: dict, initial_fund: float = 1000000.0
    ):
        self.holdout_df = holdout_df
        self.ticker_df = holdout_df.drop(columns="ES3.SI", axis=1)
        self.benchmark_df = holdout_df["ES3.SI"]
        self.weights = weights
        self.initial_fund = initial_fund

    def evaluate_portfolio_performance_on_relative_returns(self):
        daily_return_df = self.ticker_df.pct_change().dropna()
        portfolio_daily_return = daily_return_df.dot(
            list(self.weights.values())
        )  # .values if its numpy array
        portfolio_sharpe_ratio_annualized = (
            portfolio_daily_return.mean() / portfolio_daily_return.std()
        ) * 252**0.5
        return portfolio_sharpe_ratio_annualized

    def calculate_fund_allocation(self):
        # Calculate the fund allocation based on the weights
        starting_price = self.ticker_df.iloc[0]
        fund_list = [weight * self.initial_fund for weight in self.weights.values()]

        # Perform element-wise division
        shares, remainder = divmod(fund_list, starting_price)

        return shares, remainder

    def calculate_daily_capital(self, shares):
        # multiply the shares by the daily price
        daily_prices = self.ticker_df
        daily_capital = daily_prices * shares
        return daily_capital

    def evaluate_portfolio_performance_dollar_amount(self, daily_capital):
        # row sum of daily capital
        portfolio_value = daily_capital.sum(axis=1)
        portfolio_daily_return = portfolio_value.pct_change().dropna()
        portfolio_sharpe_ratio_annualized = (
            portfolio_daily_return.mean() / portfolio_daily_return.std()
        ) * 252**0.5
        return portfolio_sharpe_ratio_annualized

    def benchmark_buy_and_hold_relative_return(self):
        benchmark_daily_return = self.benchmark_df.pct_change().dropna()
        benchmark_sharpe_ratio_annualized = (
            benchmark_daily_return.mean() / benchmark_daily_return.std()
        ) * 252**0.5

        return benchmark_sharpe_ratio_annualized

    def benchmark_buy_and_hold_dollar_amount(self):
        starting_price = self.benchmark_df.iloc[0]
        fund = self.initial_fund
        shares, remainder = divmod(fund, starting_price)
        daily_prices = self.benchmark_df
        daily_capital = daily_prices * shares
        portfolio_value = daily_capital
        portfolio_daily_return = portfolio_value.pct_change().dropna()
        portfolio_sharpe_ratio_annualized = (
            portfolio_daily_return.mean() / portfolio_daily_return.std()
        ) * 252**0.5
        return portfolio_sharpe_ratio_annualized


# Approach A focuses on relative returns (percentage changes), which can be skewed by the scaling effect.
# Approach B considers absolute dollar amounts invested, accounting for the scaling effect.
# Consequently, Approach B tends to provide a more balanced representation of the portfolio’s performance, especially when dealing with stocks of varying prices.
# In summary, while Approach A is simpler, Approach B provides a more accurate picture by incorporating the scaling effect and considering the actual dollar amounts invested in each stock. The choice between the two approaches depends on the investor’s preferences and practical considerations


if __name__ == "__main__":
    hold_out_df = (
        pd.read_csv("./data/price.csv").query("Date>= '2023-01-01'").set_index("Date")
    )
    weights = np.array(
        [
            0.15957085422175335,
            0.14090612359267582,
            0.000905568777916455,
            0.00041470017073928864,
            0.19502573990349809,
            0.05271553196650697,
            0.05633597182075772,
            0.14610042995712078,
            0.11604090698936258,
            0.09249270883259227,
            0.023232634534131238,
            0.01625882923294569,
        ]
    )

    backtester = Backtest(hold_out_df, weights)
    sharpe_ratio = backtester.evaluate_portfolio_performance_on_relative_returns()

    print(f"Sharpe Ratio: {sharpe_ratio}")
