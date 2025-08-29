import ta
import numpy as np
import pandas as pd

from logger import trading_logger

core_feature_cols = [
    'adx', 'plus_di', 'minus_di',
    'cmf',
    'mfi',
    'vwap',
    'macd', 'macd_signal', 'macd_diff',
    'rsi',
    'kdj_k', 'kdj_d'
]
feature_cols = [
    'ichimoku_a', 'ichimoku_b',
    'bb_lower', 'bb_upper', 'bb_middle', 'bb_width',
    'ema_short', 'ema_mid', 'ema_long',
    'typical_price',
    'atr',
    'sma_short', 'sma_mid',
    'donchian_upper', 'donchian_lower',
    'tema'
]
feature_cols_all = (
    [f'4h_{col}' for col in feature_cols] +
    [f'1h_{col}_mean' for col in core_feature_cols] +
    [f'1h_{col}_std' for col in core_feature_cols] +
    [f'daily_{col}' for col in core_feature_cols]
)

# 不同时间框架的参数配置
PARAMS = {
    '1H': {
        'ema': [5, 10, 20],  # 短窗口，快速响应
        'macd': {'window_fast': 6, 'window_slow': 13, 'window_sign': 5},
        'rsi': {'window': 7},
        'cci': {'window': 10},
        'cmf': {'window': 10},
        'mfi': {'window': 10},
        'vwap': {'window': 20},
        'atr': {'window': 7},
        'bb': {'window': 10, 'window_dev': 1.5},
        'adx': {'window': 7},
        'kdj': {'window': 7, 'smooth_window': 3},
        'ichimoku': {'window1': 5, 'window2': 13, 'window3': 26},
        'psar': {'step': 0.02, 'max_step': 0.2},
        'sma': [3, 10],
        'tema': {'window': 5},
        'donchian': {'window': 10}
    },
    '4H': {
        'ema': [5, 13, 55],  # 中期窗口，平衡敏感性和稳定性
        'macd': {'window_fast': 8, 'window_slow': 17, 'window_sign': 5},
        'rsi': {'window': 14},
        'cci': {'window': 20},
        'cmf': {'window': 20},
        'mfi': {'window': 14},
        'vwap': {'window': 30},
        'atr': {'window': 14},
        'bb': {'window': 20, 'window_dev': 2.0},
        'adx': {'window': 14},
        'kdj': {'window': 14, 'smooth_window': 3},
        'ichimoku': {'window1': 9, 'window2': 26, 'window3': 52},
        'psar': {'step': 0.02, 'max_step': 0.2},
        'sma': [5, 20],
        'tema': {'window': 10},
        'donchian': {'window': 20}
    },
    'D': {
        'ema': [10, 20, 100],  # 长窗口，平滑噪声
        'macd': {'window_fast': 12, 'window_slow': 26, 'window_sign': 9},
        'rsi': {'window': 21},
        'cci': {'window': 30},
        'cmf': {'window': 30},
        'mfi': {'window': 21},
        'vwap': {'window': 50},
        'atr': {'window': 21},
        'bb': {'window': 30, 'window_dev': 2.5},
        'adx': {'window': 21},
        'kdj': {'window': 21, 'smooth_window': 5},
        'ichimoku': {'window1': 12, 'window2': 36, 'window3': 72},
        'psar': {'step': 0.02, 'max_step': 0.2},
        'sma': [10, 50],  # D：长窗口，平滑噪声
        'tema': {'window': 20},  # D：较长TEMA，捕捉长期趋势
        'donchian': {'window': 50}  # D：长Donchian窗口，适合长期通道
    }
}

def calculate_tema(close, window):
    """手动计算 TEMA"""
    ema1 = ta.trend.ema_indicator(close, window=window)
    ema2 = ta.trend.ema_indicator(ema1, window=window)
    ema3 = ta.trend.ema_indicator(ema2, window=window)
    return 3 * ema1 - 3 * ema2 + ema3

# 特征生成函数
def create_features(df, timeframe='4H'):
    try:
        df = df.copy()
        params = PARAMS[timeframe]

        # 价格指标
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=params['ema'][0])
        df['ema_mid'] = ta.trend.ema_indicator(df['close'], window=params['ema'][1])
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=params['ema'][2])
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['sma_short'] = ta.trend.sma_indicator(df['close'], window=params['sma'][0])
        df['sma_mid'] = ta.trend.sma_indicator(df['close'], window=params['sma'][1])
        df['tema'] = calculate_tema(df['close'], window=params['tema']['window'])
        donchian = ta.volatility.DonchianChannel(
            high=df['high'], low=df['low'], close=df['close'],
            window=params['donchian']['window']
        )
        df['donchian_upper'] = donchian.donchian_channel_hband()
        df['donchian_lower'] = donchian.donchian_channel_lband()

        # 趋势强度指标
        df['psar'] = ta.trend.PSARIndicator(
            high=df['high'], low=df['low'], close=df['close'],
            step=params['psar']['step'], max_step=params['psar']['max_step']
        ).psar()
        adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=params['adx']['window'])
        df['adx'] = adx_indicator.adx()
        df['plus_di'] = adx_indicator.adx_pos()
        df['minus_di'] = adx_indicator.adx_neg()
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['high'], low=df['low'],
            window1=params['ichimoku']['window1'],
            window2=params['ichimoku']['window2'],
            window3=params['ichimoku']['window3']
        )
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        # df['cloud_trend'] = (df['ichimoku_a'] > df['ichimoku_b']).astype(int)
        # df['bull_signal'] = (df['close'] > df['ichimoku_a']) & (df['close'] > df['ichimoku_b']) & (df['cloud_trend'] == 1)
        # df['bear_signal'] = (df['close'] < df['ichimoku_a']) & (df['close'] < df['ichimoku_b']) & (df['cloud_trend'] == 0)

        # 动量指标
        macd = ta.trend.MACD(
            df['close'],
            window_fast=params['macd']['window_fast'],
            window_slow=params['macd']['window_slow'],
            window_sign=params['macd']['window_sign']
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['cci'] = (
            (df['typical_price'] - df['typical_price'].rolling(params['cci']['window']).mean()) /
            (0.015 * df['typical_price'].rolling(params['cci']['window']).std())
        )
        df['rsi'] = ta.momentum.rsi(df['close'], window=params['rsi']['window'])
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'],
            window=params['kdj']['window'],
            smooth_window=params['kdj']['smooth_window']
        )
        df['kdj_k'] = stoch.stoch()
        df['kdj_d'] = stoch.stoch_signal()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        # 成交量指标
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume'],
            window=params['cmf']['window']
        ).chaikin_money_flow()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume'],
            window=params['vwap']['window']
        ).volume_weighted_average_price()
        df['mfi'] = ta.volume.money_flow_index(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume'],
            window=params['mfi']['window']
        )

        # 波动性指标
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'],
            window=params['atr']['window']
        )
        df['bb_upper'] = ta.volatility.bollinger_hband(
            df['close'], window=params['bb']['window'], window_dev=params['bb']['window_dev']
        )
        df['bb_middle'] = ta.volatility.bollinger_mavg(
            df['close'], window=params['bb']['window']
        )
        df['bb_lower'] = ta.volatility.bollinger_lband(
            df['close'], window=params['bb']['window'], window_dev=params['bb']['window_dev']
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

        # 数据清洗
        df = df[df['volume'] > 0]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        trading_logger.info(f"特征生成后数据形状: {df.shape}")
        print('是否存在NaN:', df.isna().any().any())
        print('是否存在正无穷:', np.isinf(df).any().any())

        return df
    except Exception as e:
        trading_logger.error(f"特征生成失败: {str(e)}")
        raise

def create_trend_label(df, pred_len=6):
    try:
        df = df.copy()
        df['future_price'] = df['close'].shift(-pred_len)
        df = df.dropna(subset=['future_price'])
        df['future_return'] = df['future_price'] / df['close'] - 1

        # 统计涨幅分布
        trading_logger.info(f"future_return 分位数: {df['future_return'].quantile([0.01, 0.05, 0.2, 0.25, 0.35, 0.65, 0.75, 0.8, 0.95, 0.99]).to_dict()}")
        trading_logger.info(f"累计回报率均值: {df['future_return'].mean():.4f}")
        trading_logger.info(f"累计回报中位数: {df['future_return'].median():.4f}")
        trading_logger.info(f"累计回报率标准差: {df['future_return'].std():.4f}")

        up_ratio = (df['future_return'] > 0.02).mean()
        down_ratio = (df['future_return'] < -0.02).mean()

        trading_logger.info(f"涨幅大于 2% 的比例: {up_ratio:.2%}")
        trading_logger.info(f"跌幅大于 2% 的比例: {down_ratio:.2%}")

        long_median = df.loc[df['future_return'] > 0.02, 'future_return'].median()
        short_median = df.loc[df['future_return'] < -0.02, 'future_return'].median()

        trading_logger.info(f"long_median中位数: {long_median:.2%}")
        trading_logger.info(f"short_median中位数: {short_median:.2%}")

        return df, long_median, short_median
    except Exception as e:
        trading_logger.error(f"create_trend_label get err: {str(e)}")
        raise

def align_timeframes(df_1h, df_4h, df_daily):
    try:
        # 为1小时线和日线添加前缀
        core_feature_cols_with_time = core_feature_cols + ['timestamp']
        df_1h = df_1h.rename(columns=lambda col: f"1h_{col}" if col in core_feature_cols_with_time else col)
        df_daily = df_daily.rename(columns=lambda col: f"daily_{col}" if col in core_feature_cols_with_time else col)
        df_4h = df_4h.rename(columns=lambda col: f"4h_{col}" if col in feature_cols else col)

        # 1小时线：聚合4个1小时K线到1个4小时K线
        df_1h['timestamp_4h'] = df_1h['1h_timestamp'].dt.floor('4h')
        agg_funcs = {f'1h_{col}': ['mean', 'std'] for col in core_feature_cols}
        df_1h_agg = df_1h.groupby('timestamp_4h').agg(agg_funcs).reset_index()
        df_1h_agg.columns = ['timestamp_4h'] + [f"{col[0]}_{col[1]}" for col in df_1h_agg.columns[1:]]

        df_daily['timestamp_4h'] = df_daily['daily_timestamp'].dt.floor('4h')
        df_daily = df_daily.drop(columns=['daily_timestamp'])

        # 合并数据集
        df_merged = pd.merge_asof(
            df_4h.sort_values('timestamp'),
            df_1h_agg.sort_values('timestamp_4h'),
            left_on='timestamp',
            right_on='timestamp_4h',
            direction='backward'
        )
        df_merged = pd.merge_asof(
            df_merged.sort_values('timestamp'),
            df_daily.sort_values('timestamp_4h'),
            left_on='timestamp',
            right_on='timestamp_4h',
            direction='backward'
        )
        df_merged = df_merged.drop(columns=['timestamp_4h_x', 'timestamp_4h_y'])
        df_merged = df_merged.dropna()

        return df_merged
    except Exception as e:
        print(f'align_timeframes get err: {e}')
        raise

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor
    import shap
    import pandas as pd

    from data import load_data

    def feature_importance(df):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(df[feature_cols], df['future_return'])

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(df[feature_cols])
        shap_importance = pd.DataFrame({
            'feature': feature_cols,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='mean_abs_shap', ascending=False)
        print(shap_importance)

    # 加载数据
    # df_1h = load_data('btc_1h.csv')
    df_4h = load_data('btc_4h.csv')
    # df_daily = load_data('btc_d.csv')

    # 生成特征
    # df_1h = create_features(df_1h, timeframe='1H')
    df_4h = create_features(df_4h, timeframe='4H')
    # df_daily = create_features(df_daily, timeframe='D')

    # 生成4小时线标签
    df_4h = create_trend_label(df_4h)
    # feature_importance(df_4h)