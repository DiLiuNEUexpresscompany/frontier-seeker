import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# 添加自定义模块路径（请根据实际情况调整）
sys.path.append('E:/neu/bigdata/frontier-seeker')

# 导入自定义模块
from stocknet.data.loader import StockDataLoader
from stocknet.feature.technical_feature_builder import TechnicalFeatureBuilder
from stocknet.feature.auto_feature_selector import AutoFeatureSelector
from stocknet.networks.graphical_lasso import FeatureEnhancedNetworkBuilder

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=== 开始测试流程 ===")
    
    # 1. 数据加载
    # 设置高价和低价股票数据目录（请根据实际情况修改）
    high_price_dir = "../../union_high_price_data_by_day_5years"  
    low_price_dir = "../../union_low_price_data_by_day_5years_async"  
    
    # 检查目录是否存在
    if not os.path.exists(high_price_dir) or not os.path.exists(low_price_dir):
        logger.error("高价或低价股票数据目录不存在，请检查路径设置。")
        sys.exit(1)
    
    logger.info("步骤1: 使用 StockDataLoader 加载股票数据")
    loader = StockDataLoader(min_data_points=200, fill_missing=True, n_jobs=4)
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    all_data, returns = loader.load_high_low_price_data(
        high_price_dir,
        low_price_dir,
        columns=columns,
        auto_returns=True
    )
    logger.info(f"加载完成 - 合并数据形状: {all_data.shape}")
    logger.info(f"高价股票数量: {len(loader.high_price_symbols)}")
    logger.info(f"低价股票数量: {len(loader.low_price_symbols)}")
    
    # 2. 技术指标构建
    logger.info("步骤2: 构建技术指标")
    feature_builder = TechnicalFeatureBuilder(
        include_groups=['momentum', 'trend', 'volatility', 'volume', 'price_transform']
    )
    data_with_features = feature_builder.build_features(
        all_data, 
        include_original=True,
        drop_na=True,
        min_periods=50
    )
    logger.info(f"技术指标构建完成 - 数据形状: {data_with_features.shape}")
    
    # 3. 特征选择（选取最重要的技术指标）
    logger.info("步骤3: 特征选择")
    feature_selector = AutoFeatureSelector(
        max_features=10,
        n_jobs=4,
        cv_folds=5
    )
    future_periods = [1, 5, 10]
    selected_data = feature_selector.select_features(
        data_with_features,
        future_periods=future_periods,
        target_col='close',
        method='auto'
    )
    logger.info(f"特征选择完成 - 选定特征数据形状: {selected_data.shape}")
    
    # 获取每只股票的特征重要性（字典格式）
    feature_importance = feature_selector.get_feature_importance()
    
    # 4. 定义股票池
    # 直接将加载的高价和低价股票分别作为高价股票池和低价股票池
    high_value_stocks = loader.high_price_symbols
    low_value_stocks = loader.low_price_symbols
    logger.info(f"高价股票池数量: {len(high_value_stocks)}")
    logger.info(f"低价股票池数量: {len(low_value_stocks)}")
    
    # 5. 构建跨层级网络（寻找低价与高价股票之间的关系）
    logger.info("步骤5: 构建跨层级网络")
    network_builder = FeatureEnhancedNetworkBuilder(selected_data, feature_importance)
    network_builder.set_stock_groups(high_value_stocks, low_value_stocks)
    
    # 直接使用所有高价和低价股票构建跨层级网络
    cross_network = network_builder.build_cross_value_network(high_value_stocks, low_value_stocks)
    logger.info(f"跨层级网络: {len(cross_network.nodes())} 个节点, {len(cross_network.edges())} 条边")
    
    # 输出部分边关系信息
    logger.info("低价与高价股票之间的关系（部分边信息）：")
    for u, v, d in list(cross_network.edges(data=True))[:10]:
        logger.info(f"  {u} - {v}, 权重: {d['weight']:.4f}")
    
    # 6. 可视化跨层级网络（可选）
    logger.info("步骤6: 可视化跨层级网络")
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(cross_network, seed=42)
    nx.draw(cross_network, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("高价与低价股票跨层级网络")
    plt.show()
    
    logger.info("=== 测试流程结束 ===")
