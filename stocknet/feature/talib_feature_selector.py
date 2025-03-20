import pandas as pd
import logging
from typing import List
from stocknet.data.feature import FeatureSelector  # 假设此处已有FeatureSelector的实现
from .technical_feature_builder import TechnicalFeatureBuilder

class TALibFeatureSelector(FeatureSelector):
    """
    基于TA-Lib的特征选择器
    
    扩展基础特征选择器，结合技术分析指标进行特征选择。
    首先构建技术分析指标，然后选择最有价值的特征。
    
    属性:
        max_features (int): 每只股票保留的最大特征数
        n_jobs (int): 并行计算的作业数
        talib_feature_builder (TechnicalFeatureBuilder): 技术指标构建器
    """
    
    def __init__(self, max_features: int = 5, n_jobs: int = 4, include_groups: List[str] = None):
        super().__init__(max_features, n_jobs)
        self.talib_feature_builder = TechnicalFeatureBuilder(include_groups=include_groups)
    
    def select_features(self, data: pd.DataFrame, method: str = 'mutual_info', target_col: str = 'close', build_talib_features: bool = True) -> pd.DataFrame:
        """
        为每只股票选择最重要的特征，包括TA-Lib技术指标
        
        参数:
            data (pd.DataFrame): 包含多个股票和多个特征的DataFrame (MultiIndex列)
            method (str): 特征选择方法 ('mutual_info', 'correlation', 'pca')
            target_col (str): 目标列名，用于评估其他特征的重要性
            build_talib_features (bool): 是否构建技术分析指标
            
        返回:
            pd.DataFrame: 只包含选择的特征的DataFrame
        """
        if data.empty:
            self.logger.error("输入数据为空")
            return data
        
        if not isinstance(data.columns, pd.MultiIndex):
            self.logger.error("输入数据列不是MultiIndex格式，无法进行特征选择")
            return data
        
        if build_talib_features:
            self.logger.info("构建TA-Lib技术分析指标...")
            data_with_indicators = self.talib_feature_builder.build_features(data, include_original=True, drop_na=True, min_periods=50)
            self.logger.info(f"技术指标构建完成，数据形状: {data_with_indicators.shape}")
        else:
            data_with_indicators = data
        
        self.logger.info("选择最重要的特征...")
        selected_data = super().select_features(data_with_indicators, method, target_col)
        return selected_data
