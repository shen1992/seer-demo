import os
import time
import logging
from datetime import datetime
import pytz

os.environ['TZ'] = 'UTC'
time.tzset()

class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        utc_time = datetime.fromtimestamp(record.created, tz=pytz.UTC)
        if datefmt:
            return utc_time.strftime(datefmt)
        return utc_time.strftime("%Y-%m-%d %H:%M:%S %Z")

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

trading_logger = logging.getLogger('trading')
trading_logger.setLevel(logging.INFO)
trading_handler = logging.FileHandler(os.path.join(log_dir, 'trading.log'))
trading_handler.setFormatter(UTCFormatter('%(asctime)s UTC - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
trading_logger.addHandler(trading_handler)
trading_logger.addHandler(logging.StreamHandler())

optuna_logger = logging.getLogger('optuna')
optuna_logger.setLevel(logging.INFO)
optuna_handler = logging.FileHandler(os.path.join(log_dir, 'backtest_optimization.log'))
optuna_handler.setFormatter(UTCFormatter('%(asctime)s UTC - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
optuna_logger.addHandler(optuna_handler)
optuna_logger.addHandler(logging.StreamHandler())