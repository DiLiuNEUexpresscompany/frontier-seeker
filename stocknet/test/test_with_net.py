import os
import sys
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

# 设置路径(根据实际情况修改)
sys.path.append('E:/neu/bigdata/frontier-seeker')

# 导入自定义模块
from stocknet.feature.auto_feature_selector import AutoFeatureSelector
from stocknet.networks.graphical_lasso import FeatureEnhancedNetworkBuilder

# 设置日志级别
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=== 开始测试多层次网络构建 ===")
    
    # 1. 加载第一阶段处理后的数据
    logger.info("步骤1: 加载第一阶段处理后的数据")
    
    # 查找最新的处理后数据文件
    processed_dir = "./processed_data"
    if not os.path.exists(processed_dir):
        logger.error(f"处理后数据目录不存在: {processed_dir}")
        logger.info("请先运行第一阶段的特征选择脚本")
        sys.exit(1)
    
    # 查找最新的特征数据和重要性数据
    selected_files = [f for f in os.listdir(processed_dir) if f.startswith("selected_features_")]
    importance_files = [f for f in os.listdir(processed_dir) if f.startswith("feature_importance_")]
    
    if not selected_files or not importance_files:
        logger.error("找不到处理后的数据文件")
        logger.info("请先运行第一阶段的特征选择脚本")
        sys.exit(1)
    
    # 获取最新文件
    selected_file = sorted(selected_files)[-1]
    importance_file = sorted(importance_files)[-1]
    
    logger.info(f"加载特征数据: {selected_file}")
    logger.info(f"加载重要性数据: {importance_file}")
    
    # 加载数据
    try:
        selected_data = pd.read_csv(os.path.join(processed_dir, selected_file), header=[0, 1], index_col=0)
        logger.info(f"特征数据形状: {selected_data.shape}")
        
        # 加载特征重要性数据
        importance_df = pd.read_csv(os.path.join(processed_dir, importance_file))
        
        # 将重要性数据转换为字典格式
        feature_importances = {}
        for symbol in importance_df['symbol'].unique():
            symbol_data = importance_df[importance_df['symbol'] == symbol]
            feature_importances[symbol] = symbol_data[['feature', 'importance']]
        
        logger.info(f"加载了 {len(feature_importances)} 只股票的特征重要性")
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        sys.exit(1)
    
    # 2. 定义高价值和低价值股票组
    logger.info("\n步骤2: 定义高价值和低价值股票组")
    
    # 获取所有股票
    all_stocks = selected_data.columns.get_level_values(0).unique()
    
    # 加载或创建股票分组信息(示例中简单按字母顺序分组)
    # 在实际应用中，可能需要根据市值、行业等因素进行分组
    sorted_stocks = sorted(all_stocks)
    half_point = len(sorted_stocks) // 2
    
    high_value_stocks = sorted_stocks[:half_point]
    low_value_stocks = sorted_stocks[half_point:]
    
    logger.info(f"高价值股票: {len(high_value_stocks)} 只")
    logger.info(f"低价值股票: {len(low_value_stocks)} 只")
    logger.info(f"高价值股票示例: {', '.join(high_value_stocks[:5])}...")
    logger.info(f"低价值股票示例: {', '.join(low_value_stocks[:5])}...")
    
    # 3. 创建网络构建器
    logger.info("\n步骤3: 创建特征增强网络构建器")
    
    network_builder = FeatureEnhancedNetworkBuilder(selected_data, feature_importances)
    network_builder.set_stock_groups(high_value_stocks, low_value_stocks)
    
    # 4. 构建内部网络
    logger.info("\n步骤4: 构建内部网络")
    
    alpha = 0.1  # Graphical Lasso正则化参数，控制网络稀疏度
    high_value_network, low_value_network = network_builder.build_feature_weighted_networks(alpha=alpha)
    
    logger.info(f"高价值网络: {len(high_value_network.nodes())} 个节点, {len(high_value_network.edges())} 条边")
    logger.info(f"低价值网络: {len(low_value_network.nodes())} 个节点, {len(low_value_network.edges())} 条边")
    
    # 5. 提取代表性股票
    logger.info("\n步骤5: 提取代表性股票")
    
    # 设置每个组提取的代表性股票数量
    repr_count = 5
    
    # 计算中心性并提取代表性股票
    try:
        high_centrality = nx.eigenvector_centrality_numpy(high_value_network, weight='weight')
        top_high_stocks = sorted(high_centrality.items(), key=lambda x: x[1], reverse=True)[:repr_count]
        high_repr_stocks = [stock for stock, _ in top_high_stocks]
        
        low_centrality = nx.eigenvector_centrality_numpy(low_value_network, weight='weight')
        top_low_stocks = sorted(low_centrality.items(), key=lambda x: x[1], reverse=True)[:repr_count]
        low_repr_stocks = [stock for stock, _ in top_low_stocks]
        
        logger.info(f"高价值代表性股票: {', '.join(high_repr_stocks)}")
        logger.info(f"低价值代表性股票: {', '.join(low_repr_stocks)}")
    except Exception as e:
        logger.error(f"提取代表性股票时出错: {e}")
        logger.info("使用度中心性作为替代方法")
        
        high_degree = dict(high_value_network.degree(weight='weight'))
        top_high_stocks = sorted(high_degree.items(), key=lambda x: x[1], reverse=True)[:repr_count]
        high_repr_stocks = [stock for stock, _ in top_high_stocks]
        
        low_degree = dict(low_value_network.degree(weight='weight'))
        top_low_stocks = sorted(low_degree.items(), key=lambda x: x[1], reverse=True)[:repr_count]
        low_repr_stocks = [stock for stock, _ in top_low_stocks]
        
        logger.info(f"高价值代表性股票(度中心性): {', '.join(high_repr_stocks)}")
        logger.info(f"低价值代表性股票(度中心性): {', '.join(low_repr_stocks)}")
    
    # 6. 构建跨价值层级网络
    logger.info("\n步骤6: 构建跨价值层级网络")
    
    cross_network = network_builder.build_cross_value_network(high_repr_stocks, low_repr_stocks)
    
    logger.info(f"跨价值网络: {len(cross_network.nodes())} 个节点, {len(cross_network.edges())} 条边")
    
    # 7. 分析网络特性
    logger.info("\n步骤7: 分析网络特性")
    
    # 计算网络度量
    high_density = nx.density(high_value_network)
    low_density = nx.density(low_value_network)
    cross_density = nx.density(cross_network)
    
    logger.info(f"高价值网络密度: {high_density:.4f}")
    logger.info(f"低价值网络密度: {low_density:.4f}")
    logger.info(f"跨价值网络密度: {cross_density:.4f}")
    
    # 分析跨价值连接
    connection_counts = {}
    for high_stock in high_repr_stocks:
        connections = []
        for low_stock in low_repr_stocks:
            if cross_network.has_edge(high_stock, low_stock):
                weight = cross_network[high_stock][low_stock]['weight']
                connections.append((low_stock, weight))
        
        connection_counts[high_stock] = connections
    
    logger.info("高价值股票与低价值股票的连接情况:")
    for high_stock, connections in connection_counts.items():
        logger.info(f"  {high_stock}: 连接了 {len(connections)} 只低价值股票")
        if connections:
            top_connection = max(connections, key=lambda x: x[1])
            logger.info(f"    最强连接: {high_stock} -- {top_connection[0]} (权重: {top_connection[1]:.4f})")
    
    # 8. 可视化网络
    logger.info("\n步骤8: 可视化网络")
    
    # 创建输出目录
    output_dir = "./network_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化跨价值网络
    plt.figure(figsize=(12, 10))
    
    # 设置节点颜色
    node_colors = ['red' if cross_network.nodes[n]['value_type'] == 'high' else 'blue' 
                 for n in cross_network.nodes()]
    
    # 设置节点大小
    node_sizes = [300 for _ in cross_network.nodes()]
    
    # 设置边宽度
    edge_weights = [d['weight'] * 3 for u, v, d in cross_network.edges(data=True)]
    
    # 设置布局
    pos = nx.spring_layout(cross_network, k=0.5, seed=42)
    
    # 绘图
    nx.draw_networkx_nodes(cross_network, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(cross_network, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(cross_network, pos, font_size=8)
    
    # 在边上标注共同特征(如果有)
    edge_labels = {}
    for u, v, d in cross_network.edges(data=True):
        if 'common_features' in d and d['common_features']:
            # 只显示第一个共同特征，避免标签过长
            first_feature = d['common_features'][0] if d['common_features'] else ""
            edge_labels[(u, v)] = first_feature
    
    nx.draw_networkx_edge_labels(cross_network, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("高价值与低价值股票跨层级网络")
    plt.axis('off')
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cross_value_network_{timestamp}.png")
    plt.savefig(output_file)
    logger.info(f"跨价值网络图表已保存到: {output_file}")
    plt.close()
    
    # 9. 保存网络数据
    logger.info("\n步骤9: 保存网络数据")
    
    # 创建保存目录
    network_dir = "./network_data"
    os.makedirs(network_dir, exist_ok=True)
    
    # 保存代表性股票列表
    repr_stocks_df = pd.DataFrame({
        'high_value_stocks': high_repr_stocks + [None] * (repr_count - len(high_repr_stocks)),
        'low_value_stocks': low_repr_stocks + [None] * (repr_count - len(low_repr_stocks))
    })
    
    repr_stocks_file = os.path.join(network_dir, f"representative_stocks_{timestamp}.csv")
    repr_stocks_df.to_csv(repr_stocks_file, index=False)
    logger.info(f"代表性股票列表已保存到: {repr_stocks_file}")
    
    # 保存跨价值连接信息
    connections = []
    for u, v, d in cross_network.edges(data=True):
        conn = {
            'high_stock': u if cross_network.nodes[u]['value_type'] == 'high' else v,
            'low_stock': v if cross_network.nodes[u]['value_type'] == 'high' else u,
            'weight': d['weight']
        }
        if 'common_features' in d:
            conn['common_features'] = ','.join(d['common_features']) if d['common_features'] else ''
        connections.append(conn)
    
    connections_df = pd.DataFrame(connections)
    connections_file = os.path.join(network_dir, f"cross_connections_{timestamp}.csv")
    connections_df.to_csv(connections_file, index=False)
    logger.info(f"跨价值连接信息已保存到: {connections_file}")
    
    logger.info("=== 多层次网络构建测试完成 ===")