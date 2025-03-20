from abc import ABC, abstractmethod
import logging
import pandas as pd

class FeatureSelector(ABC):
    """
    特征选择器基类
    
    所有特征选择器的抽象基类，定义通用接口和方法。
    """
    def __init__(self, max_features: int, n_jobs: int = 1):
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def select_features(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """选择最重要的特征"""
        pass