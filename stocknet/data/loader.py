import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class StockDataLoader:
    """
    股票数据加载器

    用于统一加载、管理和预处理股票数据。

    属性:
        min_data_points (int): 每只股票最少数据点要求
        fill_missing (bool): 是否自动填充缺失值
        n_jobs (int): 并行加载作业数
    """
    
    def __init__(
        self,
        min_data_points: int = 252,
        fill_missing: bool = True,
        n_jobs: int = 4
    ):
        self.min_data_points = min_data_points
        self.fill_missing = fill_missing
        self.n_jobs = n_jobs
        self.logger = self._setup_logger()
        
        # 数据存储
        self.high_price_symbols = []
        self.low_price_symbols = []
        self.high_price_data = pd.DataFrame()
        self.low_price_data = pd.DataFrame()
        self.all_data = pd.DataFrame()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('StockDataLoader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _process_file(
        self,
        file_path: str,
        columns: List[str],
        symbol_from_filename: bool,
        symbol_prefix: str,
        symbol_suffix: str
    ) -> Optional[Tuple[str, List[pd.Series]]]:
        """
        通用文件处理函数，用于加载单个股票数据文件

        参数:
            file_path (str): 文件路径
            columns (List[str]): 需要加载的列名
            symbol_from_filename (bool): 是否从文件名中提取股票代码
            symbol_prefix (str): 股票代码前缀
            symbol_suffix (str): 股票代码后缀

        返回:
            Optional[Tuple[str, List[pd.Series]]]: 成功时返回股票代码和数据序列列表，否则返回 None
        """
        try:
            df = pd.read_csv(file_path, usecols=columns)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if len(df) < self.min_data_points:
                return None

            file_name = os.path.basename(file_path)
            if symbol_from_filename:
                symbol = file_name.split("_")[0]
            else:
                symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else file_name.split(".")[0]
            symbol = f"{symbol_prefix}{symbol}{symbol_suffix}"
            
            series_list = []
            if len(columns) == 2 and 'timestamp' in columns:
                data_col = next(col for col in columns if col != 'timestamp')
                series = df.set_index('timestamp')[data_col]
                series.name = symbol
                series_list.append(series)
            else:
                df = df.set_index('timestamp')
                for col in df.columns:
                    series = df[col]
                    series.name = (symbol, col)
                    series_list.append(series)
            return symbol, series_list
        except Exception as e:
            self.logger.error(f"处理 {file_path} 时出错: {e}")
            return None

    def load_data(
        self, 
        directory: str, 
        is_high_price: bool = False,
        columns: List[str] = ['timestamp', 'close'],
        symbol_from_filename: bool = True,
        symbol_prefix: str = "",
        symbol_suffix: str = ""
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        加载股票数据

        参数:
            directory (str): 股票数据存放目录
            is_high_price (bool): 是否为高价股票
            columns (List[str]): 需要加载的列名
            symbol_from_filename (bool): 是否从文件名中提取股票代码
            symbol_prefix (str): 股票代码前缀
            symbol_suffix (str): 股票代码后缀
            
        返回:
            tuple: (股票代码列表, 包含所有股票数据的 DataFrame)
        """
        symbols = []
        stock_series_list = []
        
        if not os.path.exists(directory):
            self.logger.error(f"目录不存在: {directory}")
            return symbols, pd.DataFrame()
        
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        use_parallel = self.n_jobs > 1 and len(files) > 10
        
        if use_parallel:
            self.logger.info(f"使用 {self.n_jobs} 个并行任务加载数据")
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._process_file,
                        os.path.join(directory, file),
                        columns,
                        symbol_from_filename,
                        symbol_prefix,
                        symbol_suffix
                    ): file for file in files
                }
                for future in tqdm(futures, desc=f"加载 {directory} 数据"):
                    try:
                        result = future.result()
                        if result:
                            symbol, series_list = result
                            symbols.append(symbol)
                            stock_series_list.extend(series_list)
                    except Exception as e:
                        file = futures[future]
                        self.logger.error(f"并行处理 {file} 时出错: {e}")
        else:
            for file in tqdm(files, desc=f"加载 {directory} 数据"):
                file_path = os.path.join(directory, file)
                result = self._process_file(file_path, columns, symbol_from_filename, symbol_prefix, symbol_suffix)
                if result:
                    symbol, series_list = result
                    symbols.append(symbol)
                    stock_series_list.extend(series_list)
        
        if not stock_series_list:
            self.logger.warning(f"未加载到任何股票数据: {directory}")
            return symbols, pd.DataFrame()
        
        data = pd.concat(stock_series_list, axis=1)
        if is_high_price:
            self.high_price_symbols = symbols
            self.high_price_data = data
        else:
            self.low_price_symbols = symbols
            self.low_price_data = data
        
        self.logger.info(f"加载 {len(symbols)} 只{('高价' if is_high_price else '低价')}股票, 数据形状: {data.shape}")
        return symbols, data
    
    def merge_high_low_data(self) -> pd.DataFrame:
        """
        合并高价和低价股票数据
        
        返回:
            pd.DataFrame: 合并后的数据框
        """
        if self.high_price_data.empty and self.low_price_data.empty:
            self.logger.error("高价和低价数据都为空，请先加载数据")
            return pd.DataFrame()
        
        self.all_data = pd.concat([self.high_price_data, self.low_price_data], axis=1)
        
        if self.fill_missing:
            min_valid = int(len(self.all_data) * 0.8)
            valid_columns = self.all_data.count() >= min_valid
            self.all_data = self.all_data.loc[:, valid_columns]
            
            if isinstance(self.all_data.columns, pd.MultiIndex):
                valid_symbols = self.all_data.columns.get_level_values(0).unique()
            else:
                valid_symbols = self.all_data.columns
            
            self.high_price_symbols = [s for s in self.high_price_symbols if s in valid_symbols]
            self.low_price_symbols = [s for s in self.low_price_symbols if s in valid_symbols]
            
            self.all_data = self.all_data.interpolate(method='time').bfill().ffill()
            self.logger.info(f"缺失值处理后数据形状: {self.all_data.shape}")
        
        self.logger.info(f"合并后的数据形状: {self.all_data.shape}")
        return self.all_data
    
    def calculate_log_returns(self, price_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算对数收益率
        
        参数:
            price_data (pd.DataFrame, optional): 股票价格数据，默认使用 all_data
            
        返回:
            pd.DataFrame: 对数收益率数据
        """
        if price_data is None:
            if self.all_data.empty:
                self.logger.error("无法计算收益率：数据为空")
                return pd.DataFrame()
            price_data = self.all_data
        
        log_returns = np.log(price_data / price_data.shift(1))
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        self.logger.info(f"对数收益率数据形状: {log_returns.shape}")
        return log_returns
    
    def load_high_low_price_data(
        self,
        high_price_dir: str,
        low_price_dir: str,
        columns: List[str] = ['timestamp', 'close'],
        auto_merge: bool = True,
        auto_returns: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        一站式加载高价和低价股票数据
        
        参数:
            high_price_dir (str): 高价股票数据目录
            low_price_dir (str): 低价股票数据目录
            columns (List[str]): 需要加载的列
            auto_merge (bool): 是否自动合并数据
            auto_returns (bool): 是否自动计算收益率
            
        返回:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: 
                (合并后的数据框, 如果 auto_returns=True 则为收益率数据框)
        """
        self.high_price_symbols, self.high_price_data = self.load_data(
            high_price_dir, is_high_price=True, columns=columns
        )
        self.low_price_symbols, self.low_price_data = self.load_data(
            low_price_dir, is_high_price=False, columns=columns
        )
        
        if auto_merge:
            self.all_data = self.merge_high_low_data()
        
        returns_data = self.calculate_log_returns() if auto_returns else None
        return self.all_data, returns_data
    
    def save_processed_data(self, output_dir: str, prefix: str = "") -> None:
        """
        保存处理后的数据
        
        参数:
            output_dir (str): 输出目录
            prefix (str): 文件名前缀
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.all_data.empty:
            file_path = os.path.join(output_dir, f"{prefix}all_data_{timestamp}.csv")
            self.all_data.to_csv(file_path)
            self.logger.info(f"合并数据已保存至: {file_path}")
        
        symbols_file = os.path.join(output_dir, f"{prefix}stock_symbols_{timestamp}.csv")
        pd.DataFrame({
            'symbol': self.high_price_symbols + self.low_price_symbols,
            'type': ['high'] * len(self.high_price_symbols) + ['low'] * len(self.low_price_symbols)
        }).to_csv(symbols_file, index=False)
        self.logger.info(f"股票列表已保存至: {symbols_file}")


def load_stock_data(directory: str, min_data_points: int = 252) -> Tuple[List[str], pd.DataFrame]:
    """
    统一加载股票数据 (兼容原始函数接口)

    Parameters:
        directory (str): 股票数据存放目录
        min_data_points (int): 每只股票最少数据点要求

    Returns:
        tuple: (股票代码列表, 包含所有股票收盘价数据的 DataFrame，
                列名为股票代码，索引为时间戳)
    """
    loader = StockDataLoader(min_data_points=min_data_points)
    return loader.load_data(directory)


if __name__ == "__main__":
    # 使用示例
    loader = StockDataLoader(min_data_points=200)
    
    # 加载高价和低价股票数据，并计算收益率
    all_data, returns = loader.load_high_low_price_data(
        "../../union_high_price_data_by_day_5years",
        "../../union_low_price_data_by_day_5years_async",
        auto_returns=True,
        columns=['timestamp', 'open' ,'close']
    )
    
    print(f"合并数据形状: {all_data.shape}")
    print(f"合并数据样子: {all_data.head()}")
    print(f"收益率数据形状: {returns.shape}")
    print(f"收益率数据样子: {returns.head()}")
    print(f"高价股票数量: {len(loader.high_price_symbols)}")
    print(f"低价股票数量: {len(loader.low_price_symbols)}")
