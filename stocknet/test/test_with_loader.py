import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append('E:/neu/bigdata/frontier-seeker')
# 导入自定义模块
from stocknet.data.loader import StockDataLoader
from stocknet.feature.technical_feature_builder import TechnicalFeatureBuilder
from stocknet.feature.auto_feature_selector import AutoFeatureSelector

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 设置数据目录（根据实际情况修改）
    high_price_dir = "../../union_high_price_data_by_day_5years"  # 高价股票数据目录
    low_price_dir = "../../union_low_price_data_by_day_5years_async"  # 低价股票数据目录
    
    # 检查目录是否存在
    if not os.path.exists(high_price_dir):
        logger.error(f"高价股票数据目录不存在: {high_price_dir}")
        logger.info("请修改脚本中的路径或创建示例数据目录")
        sample_dir = "./sample_data"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            high_price_dir = os.path.join(sample_dir, "high_price")
            low_price_dir = os.path.join(sample_dir, "low_price")
            os.makedirs(high_price_dir, exist_ok=True)
            os.makedirs(low_price_dir, exist_ok=True)
            logger.info(f"已创建示例数据目录: {sample_dir}")
            logger.info("请在运行脚本前先放入样本数据")
        sys.exit(1)
    
    logger.info("=== 开始测试特征构建和选择 ===")
    
    # 1. 使用StockDataLoader加载股票数据
    logger.info("步骤1: 使用StockDataLoader加载股票数据")
    loader = StockDataLoader(min_data_points=200, fill_missing=True, n_jobs=4)
    
    # 加载高价和低价股票数据，选择需要的列
    # 这里选择的列包括：时间戳、开盘价、收盘价和成交量（如果有的话）
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    all_data, returns = loader.load_high_low_price_data(
        high_price_dir,
        low_price_dir,
        columns=columns,
        auto_returns=True
    )
    
    # 打印数据统计信息
    logger.info(f"加载完成 - 合并数据形状: {all_data.shape}")
    logger.info(f"高价股票数量: {len(loader.high_price_symbols)}")
    logger.info(f"低价股票数量: {len(loader.low_price_symbols)}")
    
    # 2. 使用TechnicalFeatureBuilder构建技术指标
    logger.info("\n步骤2: 使用TechnicalFeatureBuilder构建技术指标")
    
    # 创建特征构建器实例
    # 可以选择包含的特征组，例如：momentum、trend、volatility、volume等
    feature_builder = TechnicalFeatureBuilder(
        include_groups=['momentum', 'trend', 'volatility', 'volume', 'price_transform']
    )
    
    # 为所有股票构建技术指标
    data_with_features = feature_builder.build_features(
        all_data, 
        include_original=True,  # 保留原始价格数据
        drop_na=True,           # 删除包含NA值的行
        min_periods=50          # 保留的最小有效数据行数
    )
    
    logger.info(f"技术指标构建完成 - 数据形状: {data_with_features.shape}")
    
    # 3. 使用AutoFeatureSelector选择最重要的特征
    logger.info("\n步骤3: 使用AutoFeatureSelector进行特征选择")
    
    # 创建特征选择器实例
    feature_selector = AutoFeatureSelector(
        max_features=10,  # 每只股票保留的最大特征数
        n_jobs=4,         # 并行计算的作业数
        cv_folds=5        # 交叉验证折数
    )
    
    # 选择特征，基于对未来多个周期收益的预测能力
    future_periods = [1, 5, 10]  # 1天、5天和10天的未来收益
    selected_data = feature_selector.select_features(
        data_with_features,
        future_periods=future_periods,
        target_col='close',
        method='auto'  # 自动选择最佳特征选择方法
    )
    
    logger.info(f"特征选择完成 - 选定特征数据形状: {selected_data.shape}")
    
    # 4. 分析特征重要性结果
    logger.info("\n步骤4: 分析特征重要性")
    
    # 获取每只股票的特征重要性
    feature_importance = feature_selector.get_feature_importance()
    
    # 输出前5只股票的重要特征
    sample_symbols = (loader.high_price_symbols + loader.low_price_symbols)[:5]
    logger.info(f"样本股票特征重要性分析 (前5只股票):")
    
    for symbol in sample_symbols:
        if symbol in feature_importance:
            importance_df = feature_importance[symbol]
            top_features = importance_df.sort_values('importance', ascending=False).head(5)
            logger.info(f"\n股票 {symbol} 的前5个重要特征:")
            for idx, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 5. 可视化特征结果
    logger.info("\n步骤5: 可视化分析")
    
    # 选择第一只股票进行可视化
    if sample_symbols:
        first_symbol = sample_symbols[0]
        
        # 创建输出目录
        output_dir = "./feature_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        if first_symbol in feature_selector.selected_features:
            selected_features = feature_selector.selected_features[first_symbol]
            
            # 创建可视化图表
            plt.figure(figsize=(14, 12))
            
            # 子图1：价格走势
            plt.subplot(3, 1, 1)
            plt.title(f"{first_symbol} 价格走势")
            if 'close' in all_data[first_symbol].columns:
                close_prices = all_data[first_symbol]['close']
                plt.plot(close_prices.index, close_prices.values)
            plt.grid(True)
            
            # 子图2：选定的技术指标
            tech_features = [f for f in selected_features if f != 'close'][:3]
            if tech_features:
                plt.subplot(3, 1, 2)
                plt.title(f"选定的技术指标")
                for feature in tech_features:
                    if feature in data_with_features[first_symbol].columns:
                        feature_data = data_with_features[first_symbol][feature]
                        plt.plot(feature_data.index, feature_data.values, label=feature)
                plt.legend()
                plt.grid(True)
            
            # 子图3：特征重要性柱状图
            plt.subplot(3, 1, 3)
            plt.title(f"{first_symbol} 特征重要性")
            if first_symbol in feature_importance:
                top_importance = feature_importance[first_symbol].sort_values('importance', ascending=False).head(10)
                plt.barh(top_importance['feature'], top_importance['importance'])
                plt.tight_layout()
                
                # 保存图表
                output_file = os.path.join(output_dir, f"{first_symbol}_feature_analysis.png")
                plt.savefig(output_file)
                logger.info(f"图表已保存到: {output_file}")
            
            plt.close()
    
    # 6. 保存处理后的数据（可选）
    logger.info("\n步骤6: 保存处理后的数据")
    result_dir = "./processed_data"
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存选定特征的数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_data_file = os.path.join(result_dir, f"selected_features_{timestamp}.csv")
    selected_data.to_csv(selected_data_file)
    logger.info(f"选定特征数据已保存到: {selected_data_file}")
    
    # 保存特征重要性信息
    importance_file = os.path.join(result_dir, f"feature_importance_{timestamp}.csv")
    all_importance = []
    for symbol, imp_df in feature_importance.items():
        imp_df['symbol'] = symbol
        all_importance.append(imp_df)
    
    if all_importance:
        pd.concat(all_importance).to_csv(importance_file, index=False)
        logger.info(f"特征重要性数据已保存到: {importance_file}")
    
    logger.info("=== 测试完成 ===")