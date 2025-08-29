import pandas as pd

from logger import trading_logger

# 数据加载函数（保持不变）
def load_data(file_path='btc_data.csv', max_rows=60000):
    try:
        df = pd.read_csv(file_path)
        trading_logger.info(f"原始数据形状: {df.shape}")
        trading_logger.info(f"原始列: {df.columns.tolist()}")
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV文件缺少必要列: timestamp, open, high, low, close, volume")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.floor('s')
        df = df[required_columns].astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        trading_logger.info(f"转换后数据类型: {df.dtypes.to_dict()}")
        nan_counts = df.isna().sum()
        if nan_counts.any():
            trading_logger.warning(f"发现 NaN 值: {nan_counts.to_dict()}")
        invalid_rows = df[df[['open', 'high', 'low', 'close', 'volume']].le(0).any(axis=1)]
        if not invalid_rows.empty:
            trading_logger.warning(f"发现 {len(invalid_rows)} 条无效数据（零或负值）:\n{invalid_rows}")
            df = df[df[['open', 'high', 'low', 'close', 'volume']].gt(0).all(axis=1)]
        df = df.dropna()
        df = df.tail(max_rows)
        if df.empty:
            raise ValueError("加载数据后为空")
        trading_logger.info(f"成功加载数据，{len(df)}条记录，最新时间: {df['timestamp'].iloc[-1]}")
        return df
    except Exception as e:
        trading_logger.error(f"数据加载失败: {str(e)}")
        raise