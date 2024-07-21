import warnings
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class CreateStockData:

    def __init__(self, observation_days: int, target_days: int, tickers: list,
                 add_technical_indicators: bool = True):
        self.observation_days = observation_days
        self.target_days = target_days
        self.tickers = tickers
        self.add_ta = add_technical_indicators



        self.start_dates = self.get_earliest_dates()
        self.process_stock_data()

    def get_earliest_dates(self):
        earliest_dates = {}
        for ticker in self.tickers:
            ticker_info = yf.Ticker(ticker)
            history = ticker_info.history(period="max")
            earliest_date = history.index.min().strftime('%Y-%m-%d')
            earliest_dates[ticker] = earliest_date
        return earliest_dates

    def process_stock_data(self):
        stock_data = self.get_stock_data()

        regression_targets = self.calculate_targets(stock_data['Close'], self.target_days)
        stock_data = stock_data.iloc[:-self.target_days]

        stock_data = self.prepare_features(stock_data, self.observation_days)

        assert stock_data.shape[0] == regression_targets.shape[0]

        stock_data.reset_index(inplace=True, drop=True)



        self.save_data(stock_data, regression_targets)


    def get_stock_data(self) -> pd.DataFrame:
        data_frames = []
        for ticker in self.tickers:
            data = self.download_stock_data(ticker)
            if self.add_ta:
                data = self.add_technical_indicators(data)
            data.index = self.convert_datetime_to_int(data.index)
            # Create a DataFrame with one-hot encoded column for this ticker
            ticker_df = pd.DataFrame(0, index=data.index, columns=self.tickers)
            ticker_df[ticker] = 1

            # Concatenate the one-hot encoded column with the data
            data = pd.concat([ticker_df,data], axis=1)
            data_frames.append(data)

        combined_data = pd.concat(data_frames, axis=0).reset_index()

        # Identify and list columns that contain only zeros and are thus non-informative
        columns_to_delete = [col for col in combined_data.columns if combined_data[col].sum() == 0]

        # Remove the identified non-informative columns from the DataFrame
        stock_data_cleaned = combined_data.drop(columns=columns_to_delete)

        return stock_data_cleaned

    def prepare_features(self, stock_data: pd.DataFrame, observation_days: int) -> pd.DataFrame:
        if observation_days == 1:
            return stock_data.iloc[observation_days:]

        combined_data = stock_data.copy()
        for day in range(observation_days, 0, -1):
            shifted_df = stock_data.shift(day)
            shifted_df.columns = [f"{col}_shifted_{day}" for col in stock_data.columns]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for col in stock_data.columns:
                    combined_data.insert(combined_data.columns.get_loc(col) + 1, f"{col}_shifted_{day}",
                                         shifted_df[f"{col}_shifted_{day}"])

        return combined_data[observation_days:]

    def calculate_targets(self, close_prices: pd.Series, future_days: int) -> pd.DataFrame:
        regression_targets = pd.DataFrame(index=close_prices.index)

        for day in range(1, future_days + 1):
            future_close = close_prices.shift(-day)

            # Calculate the  change for the regression target
            regression_targets[f"Change_{day}-Day"] = (future_close/close_prices-1)

        return regression_targets.iloc[self.observation_days:-self.target_days]

    def save_data(self, features: pd.DataFrame, regress_targets: pd.DataFrame) -> int:


        features_path = 'us_stock_index_feats_indicators.csv'
        regress_targets_path = 'us_stock_index_targets_indicators.csv'

        # Initialize StandardScaler
        scaler = StandardScaler()

        # Fit and transform the data
        normalized_data = scaler.fit_transform(features)

        # Create a new DataFrame with the scaled data and original column names
        features = pd.DataFrame(normalized_data, columns=features.columns)

        features.fillna(0).to_csv(features_path, index=False)
        regress_targets.fillna(0).to_csv(regress_targets_path, index=False)

        return 0

    def decompose_signals(self,PCA_check:bool, TSNE_check:bool):
        pca = PCA(n_components=0.95)  # This will keep 95% of the variance

        # Fit and transform the normalized data
        pca_features = pca.fit_transform(normalized_data)

        # Create a new DataFrame with the PCA-transformed data
        # Note: PCA components don't have inherent meaningful names, so we'll use generic column names
        pca_column_names = [f'PC_{i + 1}' for i in range(pca_features.shape[1])]
        features_pca = pd.DataFrame(pca_features, columns=pca_column_names)

    def download_stock_data(self, symbol: str) -> pd.DataFrame:
        return yf.download(symbol, start=self.start_dates[symbol], end=None)

    @staticmethod
    def add_technical_indicators(df):
        df_cleaned = df.fillna(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_with_indicators = add_all_ta_features(
                df_cleaned, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        return df_with_indicators

    @staticmethod
    def convert_datetime_to_int(dates):
        return dates.astype('int64') // 10 ** 9


if __name__ == '__main__':

    # All
    tickers = ["^GSPC"]
    St = CreateStockData(1, 10, tickers, add_technical_indicators=True)