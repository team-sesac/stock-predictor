from .preprocessor import Preprocessor
import gc
from itertools import combinations
from numba import njit, prange
import pandas as pd
import polars as pl
import numpy as np

class ExplainedOptiverProcessor(Preprocessor):
    
    def name(self) -> str:
        return "explaind optiver"
    
    def execute_x(self, data, target=None):
        if target is not None:
            data = data.dropna(subset=[target])
        data = data.reset_index(drop=True)
        data = data.ffill().fillna(0)
        self._set_data(data)
        data = self._generate_all_features(data, target)
        return data
    
    def execute_y(self, data, target):
        data = data.dropna(subset=[target])
        return data[target]
    
    def _set_data(self, df):
        self.global_stock_id_feats = {
            "median_size": df.groupby("stock_id")["bid_size"].median() + df.groupby("stock_id")["ask_size"].median(),
            "std_size": df.groupby("stock_id")["bid_size"].std() + df.groupby("stock_id")["ask_size"].std(),
            "ptp_size": df.groupby("stock_id")["bid_size"].max() - df.groupby("stock_id")["bid_size"].min(),
            "median_price": df.groupby("stock_id")["bid_price"].median() + df.groupby("stock_id")["ask_price"].median(),
            "std_price": df.groupby("stock_id")["bid_price"].std() + df.groupby("stock_id")["ask_price"].std(),
            "ptp_price": df.groupby("stock_id")["bid_price"].max() - df.groupby("stock_id")["ask_price"].min(),
        }
        self.weights = [
            0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
            0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
            0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
            0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
            0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
            0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
            0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
            0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
            0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
            0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
            0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
            0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
            0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
            0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
            0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
            0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
            0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
        ]
        self.weights = {int(k):v for k,v in enumerate(self.weights)}
    
    def _generate_all_features(self, data, target):
        # cols = [ c for c in df.columns if c not in ["row_id", "time_id", "target"] ]
        # df = df[cols]
        drop_labels = ["row_id", "time_id"]
        if target is not None:
            drop_labels.append(target)
        data = data.drop(labels=drop_labels, axis=1)
        # df = self._imbalance_features(df)
        # gc.collect()
        # df = self._other_features(df)
        # gc.collect()
        # feature_name = [ i for i in df.columns if i not in ["row_id", "time_id", "date_id", "target"] ]
        data = data.drop(labels=["date_id"], axis=1)
        return data
        
    
    def _imbalance_features(self, df):
        prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
        sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

        df["volume"] = df.eval("ask_size + bid_size")
        df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
        df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
        df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
        df["size_imbalance"] = df.eval("bid_size / ask_size")

        for c in combinations(prices, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

        for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
            triplet_feature = self._calculate_triplet_imbalance_numba(c, df)
            df[triplet_feature.columns] = triplet_feature.values

        df["stock_weights"] = df["stock_id"].map(self.weights)
        df["weighted_wap"] = df["stock_weights"] * df["wap"]
        df['wap_momentum'] = df.groupby('stock_id')['weighted_wap'].pct_change(periods=6)

        df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
        df["price_spread"] = df["ask_price"] - df["bid_price"]
        df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
        df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
        df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
        df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])

        df['spread_depth_ratio'] = (df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])
        df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        df['micro_price'] = ((df['bid_price'] * df['ask_size']) + (df['ask_price'] * df['bid_size'])) / (df['bid_size'] + df['ask_size'])
        df['relative_spread'] = (df['ask_price'] - df['bid_price']) / df['wap']

        for func in ["mean", "std", "skew", "kurt"]:
            df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
            df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)


        for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
            for window in [1,3,5,10]:
                df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
                df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)

        for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'weighted_wap','price_spread']:
            for window in [1,3,5,10]:
                df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

        for window in [3,5,10]:
            df[f'price_change_diff_{window}'] = df[f'bid_price_diff_{window}'] - df[f'ask_price_diff_{window}']
            df[f'size_change_diff_{window}'] = df[f'bid_size_diff_{window}'] - df[f'ask_size_diff_{window}']

        pl_df = pl.from_pandas(df)

        windows = [3, 5, 10]
        columns = ['ask_price', 'bid_price', 'ask_size', 'bid_size']

        group = ["stock_id"]
        expressions = []

        for window in windows:
            for col in columns:
                rolling_mean_expr = (
                    pl.col(f"{col}_diff_{window}")
                    .rolling_mean(window)
                    .over(group)
                    .alias(f'rolling_diff_{col}_{window}')
                )

                rolling_std_expr = (
                    pl.col(f"{col}_diff_{window}")
                    .rolling_std(window)
                    .over(group)
                    .alias(f'rolling_std_diff_{col}_{window}')
                )

                expressions.append(rolling_mean_expr)
                expressions.append(rolling_std_expr)

        lazy_df = pl_df.lazy().with_columns(expressions)

        pl_df = lazy_df.collect()

        df = pl_df.to_pandas()
        gc.collect()

        df['mid_price*volume'] = df['mid_price_movement'] * df['volume']
        df['harmonic_imbalance'] = df.eval('2 / ((1 / bid_size) + (1 / ask_size))')

        for col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)

        return df
    
    def _other_features(self, df):
        df["dow"] = df["date_id"] % 5  # Day of the week
        df["seconds"] = df["seconds_in_bucket"] % 60  
        df["minute"] = df["seconds_in_bucket"] // 60  
        df['time_to_market_close'] = 540 - df['seconds_in_bucket']

        for key, value in self.global_stock_id_feats.items():
            df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

        return df
        
    def _calculate_triplet_imbalance_numba(self, price, df):
        df_values = df[price].values
        comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
        features_array = self._compute_triplet_imbalance(df_values, comb_indices)
        columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
        features = pd.DataFrame(features_array, columns=columns)
        return features
    
    def _compute_triplet_imbalance(self, df_values, comb_indices):
        return self._compute_triplet_imbalance_static(df_values, comb_indices)
    
    @staticmethod
    @njit(parallel=True)
    def _compute_triplet_imbalance_static(df_values, comb_indices):
        num_rows = df_values.shape[0]
        num_combinations = len(comb_indices)
        imbalance_features = np.empty((num_rows, num_combinations))
        for i in prange(num_combinations):
            a, b, c = comb_indices[i]
            for j in range(num_rows):
                max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
                min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
                mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
                
                if mid_val == min_val:
                    imbalance_features[j, i] = np.nan
                else:
                    imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

        return imbalance_features

    def _calculate_triplet_imbalance_numba(self, price, df):
        df_values = df[price].values
        comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
        features_array = self._compute_triplet_imbalance(df_values, comb_indices)
        columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
        features = pd.DataFrame(features_array, columns=columns)
        return features