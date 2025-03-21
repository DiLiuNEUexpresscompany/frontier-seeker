import numpy as np
import pandas as pd
import talib
import warnings
import logging
from tqdm import tqdm

class TechnicalFeatureBuilder:
    """
    技术分析特征构建器
    
    使用TA-Lib库构建常见的技术分析指标，为股票数据增加特征维度。
    
    属性:
        ohlcv_columns (Dict): OHLCV列名映射
        feature_groups (Dict): 特征分组配置
    """
    
    def __init__(self, ohlcv_columns: dict = None, include_groups: list = None):
        """
        初始化技术分析特征构建器
        
        参数:
            ohlcv_columns (dict): 自定义OHLCV列名映射
            include_groups (list): 要包含的特征组，默认全部
        """
        self.ohlcv_columns = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        if ohlcv_columns:
            self.ohlcv_columns.update(ohlcv_columns)
        
        self.feature_groups = {
            'momentum': self._build_momentum_indicators,
            'trend': self._build_trend_indicators,
            'volatility': self._build_volatility_indicators,
            'volume': self._build_volume_indicators,
            'cycle': self._build_cycle_indicators,
            'pattern': self._build_pattern_recognition,
            'statistic': self._build_statistic_indicators,
            'price_transform': self._build_price_transforms,
            'regression': self._build_regression_indicators,  # 新增线性回归指标组
            'directional': self._build_directional_indicators  # 新增方向性指标组
        }
        
        if include_groups:
            self.feature_groups = {k: v for k, v in self.feature_groups.items() if k in include_groups}
        
        self.logger = logging.getLogger('TechnicalFeatureBuilder')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def build_features(self, data: pd.DataFrame, include_original: bool = True, drop_na: bool = True, min_periods: int = 50) -> pd.DataFrame:
        """
        为股票数据构建技术分析特征
        
        参数:
            data (pd.DataFrame): 股票价格数据，格式可以是普通DataFrame或MultiIndex DataFrame
            include_original (bool): 是否在结果中包含原始价格数据
            drop_na (bool): 是否删除包含NA值的行
            min_periods (int): 保留的最小有效数据行数
            
        返回:
            pd.DataFrame: 包含原始数据和技术指标的DataFrame
        """
        if isinstance(data.columns, pd.MultiIndex):
            return self._build_features_multi_index(data, include_original, drop_na, min_periods)
        else:
            return self._build_features_single_stock(data, include_original, drop_na, min_periods)
    
    def _build_features_single_stock(self, data: pd.DataFrame, include_original: bool, drop_na: bool, min_periods: int) -> pd.DataFrame:
        df = data.copy()
        missing_cols = [col for col in ['open', 'high', 'low', 'close'] if self.ohlcv_columns[col] not in df.columns]
        if missing_cols:
            self.logger.warning(f"数据中缺少必要的列: {missing_cols}")
            if 'close' in missing_cols:
                self.logger.error("缺少收盘价列，无法构建特征")
                return df
        inputs = {}
        for key, column in self.ohlcv_columns.items():
            if column in df.columns:
                inputs[key] = df[column].values
            elif key == 'volume':
                inputs[key] = np.ones(len(df))
        
        features = {}
        for group_name, build_func in self.feature_groups.items():
            try:
                group_features = build_func(inputs)
                features.update(group_features)
                self.logger.info(f"成功构建 {group_name} 组特征: {len(group_features)} 个")
            except Exception as e:
                self.logger.error(f"构建 {group_name} 组特征时出错: {e}")
        
        features_df = pd.DataFrame(features, index=df.index)
        result = pd.concat([df, features_df], axis=1) if include_original else features_df
        
        if drop_na:
            original_len = len(result)
            result = result.dropna()
            dropped_rows = original_len - len(result)
            if len(result) < min_periods:
                self.logger.warning(f"删除NaN后数据行数 ({len(result)}) 少于要求 ({min_periods})，将填充NaN")
                result = pd.concat([df, features_df], axis=1).fillna(method='ffill').fillna(method='bfill')
            else:
                self.logger.info(f"删除了 {dropped_rows} 行包含NaN值的数据")
        return result

    def _build_features_multi_index(self, data: pd.DataFrame, include_original: bool, drop_na: bool, min_periods: int) -> pd.DataFrame:
        stocks = data.columns.get_level_values(0).unique()
        self.logger.info(f"为 {len(stocks)} 只股票构建技术指标")
        all_features_dfs = []
        for stock in tqdm(stocks, desc="构建技术指标"):
            try:
                stock_data = data[stock].copy()
                stock_features = self._build_features_single_stock(stock_data, include_original, False, min_periods)
                stock_features.columns = pd.MultiIndex.from_product([[stock], stock_features.columns], names=['stock', 'feature'])
                all_features_dfs.append(stock_features)
            except Exception as e:
                self.logger.error(f"处理股票 {stock} 时出错: {e}")
        if not all_features_dfs:
            self.logger.error("未能为任何股票构建特征")
            return data
        result = pd.concat(all_features_dfs, axis=1)
        if drop_na:
            original_len = len(result)
            result = result.dropna()
            dropped_rows = original_len - len(result)
            if len(result) < min_periods:
                self.logger.warning(f"删除NaN后数据行数 ({len(result)}) 少于要求 ({min_periods})，将填充NaN")
                result = pd.concat(all_features_dfs, axis=1).fillna(method='ffill').fillna(method='bfill')
            else:
                self.logger.info(f"删除了 {dropped_rows} 行包含NaN值的数据")
        return result

    def _build_momentum_indicators(self, inputs: dict) -> dict:
        features = {}
        for period in [6, 14, 24]:
            features[f'RSI_{period}'] = talib.RSI(inputs['close'], timeperiod=period)
        macd, macd_signal, macd_hist = talib.MACD(inputs['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        features['MACD'] = macd
        features['MACD_SIGNAL'] = macd_signal
        features['MACD_HIST'] = macd_hist
        slowk, slowd = talib.STOCH(inputs['high'], inputs['low'], inputs['close'],
                                    fastk_period=5, slowk_period=3, slowk_matype=0,
                                    slowd_period=3, slowd_matype=0)
        features['STOCH_K'] = slowk
        features['STOCH_D'] = slowd
        
        # 新增: 快速随机指标
        fastk, fastd = talib.STOCHF(inputs['high'], inputs['low'], inputs['close'],
                                    fastk_period=5, fastd_period=3, fastd_matype=0)
        features['STOCHF_K'] = fastk
        features['STOCHF_D'] = fastd
        
        # 新增: 随机RSI
        for period in [14]:
            stoch_rsi_k, stoch_rsi_d = talib.STOCHRSI(inputs['close'], timeperiod=period,
                                                      fastk_period=5, fastd_period=3, fastd_matype=0)
            features[f'STOCHRSI_K_{period}'] = stoch_rsi_k
            features[f'STOCHRSI_D_{period}'] = stoch_rsi_d
        
        for period in [10, 20]:
            features[f'ROC_{period}'] = talib.ROC(inputs['close'], timeperiod=period)
            
            # 新增: 变化率百分比
            features[f'ROCP_{period}'] = talib.ROCP(inputs['close'], timeperiod=period)
            
            # 新增: 变化率比率
            features[f'ROCR_{period}'] = talib.ROCR(inputs['close'], timeperiod=period)
            
            # 新增: 变化率比率100
            features[f'ROCR100_{period}'] = talib.ROCR100(inputs['close'], timeperiod=period)
        
        for period in [14, 20]:
            features[f'CCI_{period}'] = talib.CCI(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
        
        features['ULTOSC'] = talib.ULTOSC(inputs['high'], inputs['low'], inputs['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        features['WILLR_14'] = talib.WILLR(inputs['high'], inputs['low'], inputs['close'], timeperiod=14)
        
        # 新增: 钱德动量振荡器
        for period in [14, 20]:
            features[f'CMO_{period}'] = talib.CMO(inputs['close'], timeperiod=period)
        
        # 新增: MOM动量
        for period in [10, 14]:
            features[f'MOM_{period}'] = talib.MOM(inputs['close'], timeperiod=period)
        
        # 新增: 百分比价格振荡器
        features['PPO'] = talib.PPO(inputs['close'], fastperiod=12, slowperiod=26, matype=0)
        
        return features

    def _build_trend_indicators(self, inputs: dict) -> dict:
        features = {}
        for period in [5, 10, 20, 50, 200]:
            features[f'SMA_{period}'] = talib.SMA(inputs['close'], timeperiod=period)
        for period in [5, 10, 20, 50, 200]:
            features[f'EMA_{period}'] = talib.EMA(inputs['close'], timeperiod=period)
            
        # 新增: 双指数移动平均线
        for period in [20, 50]:
            features[f'DEMA_{period}'] = talib.DEMA(inputs['close'], timeperiod=period)
            
        # 新增: 三角移动平均线
        for period in [20, 30]:
            features[f'TRIMA_{period}'] = talib.TRIMA(inputs['close'], timeperiod=period)
            
        # 新增: 加权移动平均线
        for period in [20, 50]:
            features[f'WMA_{period}'] = talib.WMA(inputs['close'], timeperiod=period)
            
        # 新增: 三重指数平滑移动平均线
        for period in [20, 30]:
            features[f'TRIX_{period}'] = talib.TRIX(inputs['close'], timeperiod=period)
            
        features['KAMA_30'] = talib.KAMA(inputs['close'], timeperiod=30)
        features['TEMA_20'] = talib.TEMA(inputs['close'], timeperiod=20)
        
        # 方向性指标现在移动到_build_directional_indicators
        
        features['APO'] = talib.APO(inputs['close'], fastperiod=12, slowperiod=26, matype=0)
        aroon_down, aroon_up = talib.AROON(inputs['high'], inputs['low'], timeperiod=14)
        features['AROON_DOWN'] = aroon_down
        features['AROON_UP'] = aroon_up
        features['AROONOSC_14'] = talib.AROONOSC(inputs['high'], inputs['low'], timeperiod=14)
        features['BOP'] = talib.BOP(inputs['open'], inputs['high'], inputs['low'], inputs['close'])
        
        # 新增: 抛物线转向
        features['SAR'] = talib.SAR(inputs['high'], inputs['low'], acceleration=0.02, maximum=0.2)
        
        # 新增: 抛物线转向扩展版
        try:
            features['SAREXT'] = talib.SAREXT(inputs['high'], inputs['low'])
        except Exception:
            pass
            
        features['SMA_5_10_RATIO'] = features['SMA_5'] / features['SMA_10']
        features['SMA_10_20_RATIO'] = features['SMA_10'] / features['SMA_20']
        features['SMA_20_50_RATIO'] = features['SMA_20'] / features['SMA_50']
        features['SMA_50_200_RATIO'] = features['SMA_50'] / features['SMA_200']
        return features

    def _build_volatility_indicators(self, inputs: dict) -> dict:
        features = {}
        for period in [7, 14, 28]:
            features[f'ATR_{period}'] = talib.ATR(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
        for period in [10, 20, 50]:
            features[f'STDDEV_{period}'] = talib.STDDEV(inputs['close'], timeperiod=period, nbdev=1)
        
        # 新增: 方差指标
        for period in [10, 20]:
            features[f'VAR_{period}'] = talib.VAR(inputs['close'], timeperiod=period, nbdev=1)
            
        for period in [20]:
            upperband, middleband, lowerband = talib.BBANDS(inputs['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
            features[f'BBANDS_UPPER_{period}'] = upperband
            features[f'BBANDS_MIDDLE_{period}'] = middleband
            features[f'BBANDS_LOWER_{period}'] = lowerband
            features[f'BBANDS_PCT_{period}'] = (inputs['close'] - lowerband) / (upperband - lowerband)
        for period in [14]:
            features[f'NATR_{period}'] = talib.NATR(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
        features['TRANGE'] = talib.TRANGE(inputs['high'], inputs['low'], inputs['close'])
        
        # 新增: Beta指标
        try:
            high_series = pd.Series(inputs['high'])
            low_series = pd.Series(inputs['low'])
            for period in [10, 20]:
                if len(high_series) >= period:
                    features[f'BETA_{period}'] = talib.BETA(inputs['high'], inputs['low'], timeperiod=period)
        except Exception:
            pass
            
        for period in [10, 20]:
            features[f'CHAIKIN_VOL_{period}'] = talib.ADOSC(inputs['high'], inputs['low'], inputs['close'], inputs['volume'], fastperiod=3, slowperiod=period)
        return features

    def _build_volume_indicators(self, inputs: dict) -> dict:
        if 'volume' not in inputs or np.all(inputs['volume'] == 1):
            return {}
        features = {}
        features['OBV'] = talib.OBV(inputs['close'], inputs['volume'])
        features['AD'] = talib.AD(inputs['high'], inputs['low'], inputs['close'], inputs['volume'])
        features['ADOSC'] = talib.ADOSC(inputs['high'], inputs['low'], inputs['close'], inputs['volume'], fastperiod=3, slowperiod=10)
        typical_price = (inputs['high'] + inputs['low'] + inputs['close']) / 3
        features['TPVR'] = typical_price / inputs['volume']
        features['MFI_14'] = talib.MFI(inputs['high'], inputs['low'], inputs['close'], inputs['volume'], timeperiod=14)
        volume_series = pd.Series(inputs['volume'])
        features['VOLUME_ROC_5'] = volume_series.pct_change(5).values
        features['VOLUME_ROC_10'] = volume_series.pct_change(10).values
        
        # 新增: 相关系数
        close_series = pd.Series(inputs['close'])
        for period in [10, 20, 30]:
            try:
                if len(close_series) >= period and len(volume_series) >= period:
                    features[f'CORREL_{period}'] = talib.CORREL(inputs['close'], inputs['volume'], timeperiod=period)
            except Exception:
                pass
        
        for period in [10, 20]:
            price_change = pd.Series(inputs['close']).pct_change(period).values
            volume_change = volume_series.pct_change(period).values
            rolling_correlation = np.zeros_like(price_change)
            for i in range(period, len(price_change)):
                if i >= period * 2:
                    price_window = price_change[i-period:i]
                    volume_window = volume_change[i-period:i]
                    mask = ~(np.isnan(price_window) | np.isnan(volume_window))
                    if np.sum(mask) > period / 2:
                        rolling_correlation[i] = np.corrcoef(price_window[mask], volume_window[mask])[0, 1]
            features[f'PRICE_VOLUME_CORR_{period}'] = rolling_correlation
        return features

    def _build_cycle_indicators(self, inputs: dict) -> dict:
        features = {}
        try:
            features['HT_DCPERIOD'] = talib.HT_DCPERIOD(inputs['close'])
        except Exception:
            pass
        try:
            features['HT_DCPHASE'] = talib.HT_DCPHASE(inputs['close'])
        except Exception:
            pass
        try:
            inphase, quadrature = talib.HT_PHASOR(inputs['close'])
            features['HT_PHASOR_INPHASE'] = inphase
            features['HT_PHASOR_QUADRATURE'] = quadrature
        except Exception:
            pass
        try:
            sine, leadsine = talib.HT_SINE(inputs['close'])
            features['HT_SINE'] = sine
            features['HT_LEADSINE'] = leadsine
        except Exception:
            pass
        try:
            features['HT_TRENDMODE'] = talib.HT_TRENDMODE(inputs['close'])
        except Exception:
            pass
        return features

    def _build_pattern_recognition(self, inputs: dict) -> dict:
        features = {}
        # 扩展K线形态识别函数列表
        common_patterns = [
            talib.CDL2CROWS,           # 两只乌鸦
            talib.CDL3BLACKCROWS,      # 三只黑乌鸦
            talib.CDL3INSIDE,          # 三内部上涨和下跌
            talib.CDL3LINESTRIKE,      # 三线打击
            talib.CDL3OUTSIDE,         # 三外部上涨和下跌
            talib.CDL3STARSINSOUTH,    # 南方三星
            talib.CDL3WHITESOLDIERS,   # 三白兵
            talib.CDLABANDONEDBABY,    # 弃婴
            talib.CDLADVANCEBLOCK,     # 大敌当前
            talib.CDLBELTHOLD,         # 腰带线
            talib.CDLBREAKAWAY,        # 突破
            talib.CDLCLOSINGMARUBOZU,  # 收盘缺影线
            talib.CDLCONCEALBABYSWALL, # 藏婴吞没
            talib.CDLCOUNTERATTACK,    # 反击线
            talib.CDLDARKCLOUDCOVER,   # 乌云盖顶
            talib.CDLDOJI,             # 十字
            talib.CDLDOJISTAR,         # 十字星
            talib.CDLDRAGONFLYDOJI,    # 蜻蜓十字
            talib.CDLENGULFING,        # 吞没形态
            talib.CDLEVENINGDOJISTAR,  # 黄昏十字星
            talib.CDLEVENINGSTAR,      # 黄昏之星
            talib.CDLGAPSIDESIDEWHITE, # 向上/下跳空并列阳线
            talib.CDLGRAVESTONEDOJI,   # 墓碑十字
            talib.CDLHAMMER,           # 锤子线
            talib.CDLHANGINGMAN,       # 上吊线
            talib.CDLHARAMI,           # 母子线
            talib.CDLHARAMICROSS,      # 十字孕线
            talib.CDLHIGHWAVE,         # 风高浪大线
            talib.CDLHIKKAKE,          # 陷阱
            talib.CDLHIKKAKEMOD,       # 修正陷阱
            talib.CDLHOMINGPIGEON,     # 家鸽
            talib.CDLIDENTICAL3CROWS,  # 三胞胎乌鸦
            talib.CDLINNECK,           # 颈内线
            talib.CDLINVERTEDHAMMER,   # 倒锤头
            talib.CDLKICKING,          # 反冲形态
            talib.CDLKICKINGBYLENGTH,  # 由较长缺影线决定的反冲形态
            talib.CDLLADDERBOTTOM,     # 梯底
            talib.CDLLONGLEGGEDDOJI,   # 长脚十字
            talib.CDLLONGLINE,         # 长线
            talib.CDLMARUBOZU,         # 光头光脚/缺影线
            talib.CDLMATCHINGLOW,      # 相同低价
            talib.CDLMATHOLD,          # 铺垫
            talib.CDLMORNINGDOJISTAR,  # 晨星十字
            talib.CDLMORNINGSTAR,      # 晨星
            talib.CDLONNECK,           # 颈上线
            talib.CDLPIERCING,         # 刺透形态
            talib.CDLRICKSHAWMAN,      # 黄包车夫
            talib.CDLRISEFALL3METHODS, # 上升/下降三法
            talib.CDLSEPARATINGLINES,  # 分离线
            talib.CDLSHOOTINGSTAR,     # 射击之星
            talib.CDLSHORTLINE,        # 短线
            talib.CDLSPINNINGTOP,      # 纺锤
            talib.CDLSTALLEDPATTERN,   # 停顿形态
            talib.CDLSTICKSANDWICH,    # 条形三明治
            talib.CDLTAKURI,           # 探水竿
            talib.CDLTASUKIGAP,        # 跳空并列阴阳线
            talib.CDLTHRUSTING,        # 插入
            talib.CDLTRISTAR,          # 三星
            talib.CDLUNIQUE3RIVER,     # 奇特三河床
            talib.CDLUPSIDEGAP2CROWS,  # 向上跳空的两只乌鸦
            talib.CDLXSIDEGAP3METHODS  # 上升/下降跳空三法
        ]
        
        for pattern_func in common_patterns:
            pattern_name = pattern_func.__name__
            try:
                features[pattern_name] = pattern_func(inputs['open'], inputs['high'], inputs['low'], inputs['close'])
            except Exception as e:
                self.logger.warning(f"计算 {pattern_name} 时出错: {e}")
        return features

    def _build_statistic_indicators(self, inputs: dict) -> dict:
        features = {}
        close_series = pd.Series(inputs['close'])
        for period in [1, 3, 5, 10, 20]:
            features[f'PRICE_CHANGE_{period}'] = close_series.pct_change(period).values
        for period in [5, 10, 20]:
            features[f'MOMENTUM_{period}'] = close_series - close_series.shift(period)
        for period in [10, 20, 50]:
            features[f'ROLLING_MEAN_{period}'] = close_series.rolling(period).mean().values
            features[f'ROLLING_MEDIAN_{period}'] = close_series.rolling(period).median().values
            features[f'ROLLING_STD_{period}'] = close_series.rolling(period).std().values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features[f'ROLLING_SKEW_{period}'] = close_series.rolling(period).skew().values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features[f'ROLLING_KURT_{period}'] = close_series.rolling(period).kurt().values
            rolling_mean = close_series.rolling(period).mean()
            rolling_std = close_series.rolling(period).std()
            features[f'Z_SCORE_{period}'] = ((close_series - rolling_mean) / rolling_std).values
        returns = close_series.pct_change().dropna()
        for period in [10, 20, 50]:
            if len(returns) >= period:
                rolling_std = returns.rolling(period).std().values
                rolling_std = np.append(np.zeros(period) + np.nan, rolling_std)
                features[f'RETURN_VOL_{period}'] = rolling_std
        for period in [5, 10, 20]:
            roll_max = pd.Series(inputs['high']).rolling(period).max()
            roll_min = pd.Series(inputs['low']).rolling(period).min()
            features[f'RANGE_{period}'] = ((roll_max - roll_min) / roll_min).values
            
        # 新增: 最大/最小值计算
        for period in [10, 20, 50]:
            features[f'MAX_{period}'] = talib.MAX(inputs['close'], timeperiod=period)
            features[f'MIN_{period}'] = talib.MIN(inputs['close'], timeperiod=period)
            
        # 新增: 最大/最小值索引
        for period in [10, 20]:
            features[f'MAXINDEX_{period}'] = talib.MAXINDEX(inputs['close'], timeperiod=period)
            features[f'MININDEX_{period}'] = talib.MININDEX(inputs['close'], timeperiod=period)
            
        # 新增: 求和指标
        for period in [10, 20]:
            features[f'SUM_{period}'] = talib.SUM(inputs['close'], timeperiod=period)
        
        return features

    def _build_price_transforms(self, inputs: dict) -> dict:
        features = {}
        try:
            features['HT_TRENDLINE'] = talib.HT_TRENDLINE(inputs['close'])
        except Exception:
            pass
        features['AVGPRICE'] = talib.AVGPRICE(inputs['open'], inputs['high'], inputs['low'], inputs['close'])
        features['WCLPRICE'] = talib.WCLPRICE(inputs['high'], inputs['low'], inputs['close'])
        features['MEDPRICE'] = talib.MEDPRICE(inputs['high'], inputs['low'])
        features['TYPPRICE'] = talib.TYPPRICE(inputs['high'], inputs['low'], inputs['close'])
        
        # 新增: 中点价格
        for period in [10, 20]:
            features[f'MIDPOINT_{period}'] = talib.MIDPOINT(inputs['close'], timeperiod=period)
            features[f'MIDPRICE_{period}'] = talib.MIDPRICE(inputs['high'], inputs['low'], timeperiod=period)
        
        try:
            mama, fama = talib.MAMA(inputs['close'], fastlimit=0.5, slowlimit=0.05)
            features['MAMA'] = mama
            features['FAMA'] = fama
        except Exception:
            pass
        
        features['T3_10'] = talib.T3(inputs['close'], timeperiod=10, vfactor=0.7)
        
        # 新增: 时间序列预测
        for period in [10, 20]:
            try:
                features[f'TSF_{period}'] = talib.TSF(inputs['close'], timeperiod=period)
            except Exception:
                pass
        
        return features
        
    def _build_regression_indicators(self, inputs: dict) -> dict:
        """构建线性回归相关指标"""
        features = {}
        
        for period in [10, 14, 20, 30]:
            try:
                # 线性回归线
                features[f'LINEARREG_{period}'] = talib.LINEARREG(inputs['close'], timeperiod=period)
                
                # 线性回归角度
                features[f'LINEARREG_ANGLE_{period}'] = talib.LINEARREG_ANGLE(inputs['close'], timeperiod=period)
                
                # 线性回归截距
                features[f'LINEARREG_INTERCEPT_{period}'] = talib.LINEARREG_INTERCEPT(inputs['close'], timeperiod=period)
                
                # 线性回归斜率
                features[f'LINEARREG_SLOPE_{period}'] = talib.LINEARREG_SLOPE(inputs['close'], timeperiod=period)
            except Exception as e:
                self.logger.warning(f"计算周期 {period} 的回归指标时出错: {e}")
                
        return features
    
    def _build_directional_indicators(self, inputs: dict) -> dict:
        """构建方向性指标"""
        features = {}
        
        for period in [14, 20]:
            try:
                # 方向性指数
                features[f'DX_{period}'] = talib.DX(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
                
                # 平均方向性指数
                features[f'ADX_{period}'] = talib.ADX(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
                
                # 平均方向性指数评级
                features[f'ADXR_{period}'] = talib.ADXR(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
                
                # 正方向指标
                features[f'PLUS_DI_{period}'] = talib.PLUS_DI(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
                
                # 负方向指标
                features[f'MINUS_DI_{period}'] = talib.MINUS_DI(inputs['high'], inputs['low'], inputs['close'], timeperiod=period)
                
                # 正方向运动
                features[f'PLUS_DM_{period}'] = talib.PLUS_DM(inputs['high'], inputs['low'], timeperiod=period)
                
                # 负方向运动
                features[f'MINUS_DM_{period}'] = talib.MINUS_DM(inputs['high'], inputs['low'], timeperiod=period)
            except Exception as e:
                self.logger.warning(f"计算周期 {period} 的方向性指标时出错: {e}")
                
        return features

    def get_feature_descriptions(self) -> dict:
        descriptions = {
            'RSI_14': '14日相对强弱指数，衡量价格变动的速度和幅度',
            'MACD': '移动平均收敛/发散指标，表示价格趋势的强度和方向',
            'MACD_SIGNAL': 'MACD信号线，通常是MACD的9日指数移动平均线',
            'MACD_HIST': 'MACD柱状图，表示MACD与其信号线之间的差值',
            'STOCH_K': '随机指标%K线，衡量收盘价在最高价和最低价范围内的位置',
            'STOCH_D': '随机指标%D线，通常是%K的3日移动平均',
            'ROC_10': '10日变化率，衡量价格相对于10日前价格的百分比变化',
            'CCI_14': '14日商品通道指数，判断价格是否偏离其统计平均值',
            'ULTOSC': '终极震荡指标，结合多个时间周期的动量指标',
            'WILLR_14': '14日威廉指标，表示收盘价与最高价之间的关系',
            # 趋势指标
            'SMA_50': '50日简单移动平均线',
            'EMA_50': '50日指数移动平均线，对近期价格赋予更高权重',
            'KAMA_30': '30日适应性移动平均线，根据噪音调整平滑度',
            'TEMA_20': '20日三重指数移动平均线，减少滞后效应',
            'ADX_14': '14日平均趋向指数，衡量趋势的强度',
            'ADXR_14': '14日平均趋向指数评级，衡量趋势强度的变化',
            'APO': '绝对价格震荡指标，类似于MACD但使用价格而非百分比',
            'AROON_DOWN': '阿隆向下指标，衡量新低点出现的时间',
            'AROON_UP': '阿隆向上指标，衡量新高点出现的时间',
            'AROONOSC_14': '14日阿隆震荡指标，AROON_UP与AROON_DOWN的差值',
            'BOP': '均势指标，衡量买卖压力的平衡',
            'SMA_5_10_RATIO': '5日均线与10日均线的比值，衡量短期趋势',
            'SMA_50_200_RATIO': '50日均线与200日均线的比值，衡量长期趋势',
            # 波动性指标
            'ATR_14': '14日真实波动幅度均值，衡量市场波动性',
            'STDDEV_20': '20日价格标准差',
            'BBANDS_UPPER_20': '20日布林带上轨（2标准差）',
            'BBANDS_MIDDLE_20': '20日布林带中轨（移动平均线）',
            'BBANDS_LOWER_20': '20日布林带下轨（-2标准差）',
            'BBANDS_PCT_20': '当前价格在布林带范围内的位置(0-1)',
            'NATR_14': '14日归一化真实波动幅度均值',
            'TRANGE': '真实波动幅度',
            'CHAIKIN_VOL_10': '10日Chaikin波动性指标',
            # 成交量指标
            'OBV': '累积能量线，累计成交量加减',
            'AD': '聚散量指标，基于价格位置加权的成交量',
            'ADOSC': '聚散量震荡指标，AD的快慢线之差',
            'TPVR': '典型价格成交量比率',
            'MFI_14': '14日资金流量指标，价量结合的RSI',
            'VOLUME_ROC_5': '5日成交量变化率',
            'PRICE_VOLUME_CORR_20': '20日价格与成交量相关性',
            # 周期指标
            'HT_DCPERIOD': '希尔伯特变换-主导周期',
            'HT_DCPHASE': '希尔伯特变换-主导周期阶段',
            'HT_PHASOR_INPHASE': '希尔伯特变换-相量分量同相',
            'HT_PHASOR_QUADRATURE': '希尔伯特变换-相量分量正交',
            'HT_SINE': '希尔伯特变换-正弦波',
            'HT_LEADSINE': '希尔伯特变换-领先正弦波',
            'HT_TRENDMODE': '希尔伯特变换-趋势模式',
            # 形态识别
            'CDL3OUTSIDE': '三外部形态',
            'CDLENGULFING': '吞噬形态',
            'CDLHAMMER': '锤子形态',
            'CDLINVERTEDHAMMER': '倒锤子形态',
            'CDLMORNINGSTAR': '早晨之星形态',
            'CDLEVENINGSTAR': '黄昏之星形态',
            'CDLHANGINGMAN': '上吊线形态',
            'CDLSHOOTINGSTAR': '流星形态',
            'CDLMARUBOZU': '光头光脚/缺影线形态',
            'CDLHARAMI': '孕线形态',
            'CDLDOJI': '十字线形态',
            # 统计指标
            'PRICE_CHANGE_5': '5日价格变化百分比',
            'MOMENTUM_10': '10日动量（当前价格与10日前价格之差）',
            'ROLLING_MEAN_20': '20日滚动均值',
            'ROLLING_STD_20': '20日滚动标准差',
            'ROLLING_SKEW_20': '20日滚动偏度',
            'ROLLING_KURT_20': '20日滚动峰度',
            'Z_SCORE_20': '20日Z得分',
            'RETURN_VOL_20': '20日收益率波动率',
            'RANGE_10': '10日价格范围比率',
            # 价格变换
            'HT_TRENDLINE': '希尔伯特变换趋势线',
            'AVGPRICE': '平均价格',
            'WCLPRICE': '加权收盘价',
            'MEDPRICE': '中间点价格',
            'TYPPRICE': '典型价格',
            'MAMA': 'MESA自适应移动平均',
            'FAMA': 'MESA自适应移动平均的跟随线',
            'T3_10': '10日三重指数移动平均线',
        }
        return descriptions
