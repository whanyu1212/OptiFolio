import time
import pandas as pd
import yfinance as yf
from typing import List
from loguru import logger


class DataLoader:

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        benchmark_ticker: str,
        interval: str = "1d",
        risk_free_rate: float = 0.0,
    ):
        """Initializes the DataLoader class

        Args:
            tickers (List[str]): A list of tickers chosen from SGX
            start_date (str): left bound of the date range (inclusive)
            end_date (str): right bound of the date range (exclusive)
            benchmark_ticker (str, optional): Benchmark ticker that is used to represent the market.
            interval (str, optional): Interval of the pricing data. Defaults to "1d".
            risk_free_rate (float, optional): risk free rate for this calculation. Defaults to 0.0.
        """

        # Input validation

        # Check that tickers is a list of strings
        if not isinstance(tickers, list) or not all(
            isinstance(ticker, str) for ticker in tickers
        ):
            raise ValueError("tickers must be a list of strings")

        # Check that start_date and end_date are strings and can be parsed into datetime objects
        try:
            pd.to_datetime(start_date)
            pd.to_datetime(end_date)
        except ValueError:
            raise ValueError("start_date and end_date must be valid date strings")

        if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
            raise ValueError("start_date must be before end_date")

        # Check that benchmark_ticker is a string
        if not isinstance(benchmark_ticker, str):
            raise ValueError("benchmark_ticker must be a string")

        # Check that interval is a string and is one of the allowed values
        # There are more intervals available in yfinance, but we will only
        # consider these three for simplicity
        if not isinstance(interval, str) or interval not in ["1d", "1wk", "1mo"]:
            raise ValueError("interval must be a string and one of '1d', '1wk', '1mo'")

        # Check that risk_free_rate is a float
        if not isinstance(risk_free_rate, float):
            raise ValueError("risk_free_rate must be a float")
        if risk_free_rate < 0 or risk_free_rate > 1:
            raise ValueError("risk_free_rate must be within the range [0, 1]")

        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.risk_free_rate = risk_free_rate
        logger.info(
            f"Initiliazing Data extraction for tickers: {self.tickers} and benchmark: {self.benchmark_ticker}"
            f"from {self.start_date} to {self.end_date}"
        )
        self.data = self.get_data()

    def get_data(self, retries=3, delay=5) -> pd.DataFrame:
        """Download data using the yfinance library
        Args:
            retries (int, optional): retry mechanism. Defaults to 3.
            delay (int, optional): time delay between consecutive retries. Defaults to 5.

        Returns:
            pd.DataFrame: Dataframe that contains each ticker's closing price
        """
        for i in range(retries):
            try:
                logger.info(
                    "Downloading data for tickers: {}",
                    self.tickers + [self.benchmark_ticker],
                )
                data = yf.download(
                    self.tickers + [self.benchmark_ticker],
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                )["Close"]
                if not data.empty:
                    logger.info("Data downloaded successfully")
                else:
                    logger.error("Data download failed.")
                return data
            except Exception as e:
                logger.error("An error occurred during data download: %s", str(e))
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying again
                    continue
                else:
                    raise  # re-raise the last exception if all retries fail

    def train_test_split(self, cut_off_date: str) -> pd.DataFrame:
        """Split the data into training and testing sets based on the cut_off_date

        Args:
            cut_off_date (str): Date to split the data

        Returns:
            pd.DataFrame: Training and testing data
        """
        if pd.to_datetime(cut_off_date) < pd.to_datetime(self.start_date):
            raise ValueError("cut_off_date must be after the start_date")
        if pd.to_datetime(cut_off_date) > pd.to_datetime(self.end_date):
            raise ValueError("cut_off_date must be before the end_date")

        train_data = self.data.loc[self.data.index <= cut_off_date]
        test_data = self.data.loc[self.data.index > cut_off_date]

        return train_data, test_data


# Sample usage

# if __name__ == "__main__":
#     dl = DataLoader(
#         tickers=[
#             "D05.SI",
#             "O39.SI",
#             "U11.SI",
#             "Z74.SI",
#             "F34.SI",
#             "C6L.SI",
#             "C38U.SI",
#             "BN4.SI",
#             "S63.SI",
#             "A17U.SI",
#             "G13.SI",
#         ],
#         start_date="2015-01-01",
#         end_date="2024-01-01",
#     )
#     # train, test = dl.train_test_split("2022-12-31")
#     print(dl.data.head())

#     dl.data.to_csv("./data/price.csv")
