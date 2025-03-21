import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler

class FeatureEnhancedNetworkBuilder:
    def __init__(self, selected_features_data, feature_importances):
        """
        构造函数
        
        参数:
            selected_features_data: AutoFeatureSelector输出的精选特征数据
            feature_importances: AutoFeatureSelector输出的特征重要性
        """
        self.data = selected_features_data
        self.feature_importances = feature_importances
        self.high_value_stocks = []  # 高价值股票列表
        self.low_value_stocks = []   # 低价值股票列表
    
    def set_stock_groups(self, high_value_stocks, low_value_stocks):
        """设置高价值和低价值股票组"""
        self.high_value_stocks = high_value_stocks
        self.low_value_stocks = low_value_stocks
    
    def build_feature_weighted_networks(self, alpha=0.05):
        """使用特征加权的Graphical Lasso构建网络"""
        # 构建高价值股票网络
        high_value_network = self._build_feature_enhanced_network(
            self.high_value_stocks, alpha, 'high_value_network')
        
        # 构建低价值股票网络
        low_value_network = self._build_feature_enhanced_network(
            self.low_value_stocks, alpha, 'low_value_network')
        
        return high_value_network, low_value_network
    
    def _build_feature_enhanced_network(self, stock_list, alpha, network_name):
        """构建基于特征增强的网络"""
        # 1. 提取这些股票的所有精选特征数据
        stock_features = {}
        for stock in stock_list:
            # 获取该股票的精选特征
            if stock in self.data.columns.get_level_values(0):
                selected_features = self.data[stock].columns.tolist()
                feature_values = self.data[stock].values
                stock_features[stock] = {
                    'features': selected_features,
                    'values': feature_values
                }
        
        # 2. 计算基于特征的相似性矩阵
        similarity_matrix = np.zeros((len(stock_list), len(stock_list)))
        for i, stock1 in enumerate(stock_list):
            for j, stock2 in enumerate(stock_list):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 对角线为1
                    continue
                
                # 计算两只股票间的特征相似度
                similarity = self._calculate_feature_similarity(stock1, stock2)
                similarity_matrix[i, j] = similarity
        
        # 3. 使用Graphical Lasso估计精确矩阵
        glasso = GraphicalLasso(alpha=alpha)
        try:
            # 标准化相似性矩阵
            scaler = StandardScaler()
            scaled_matrix = scaler.fit_transform(similarity_matrix)
            
            glasso.fit(scaled_matrix)
            precision_matrix = glasso.precision_
        except:
            # 如果Glasso失败，使用简单阈值方法
            precision_matrix = np.copy(similarity_matrix)
            # 设置一个阈值，低于该值的连接视为0
            precision_matrix[precision_matrix < 0.3] = 0
        
        # 4. 构建网络
        G = nx.Graph(name=network_name)
        
        # 添加节点及其特征属性
        for stock in stock_list:
            # 获取该股票的重要特征
            if stock in self.feature_importances:
                top_features = self.feature_importances[stock].sort_values('importance', ascending=False).head(3)
                top_feature_names = top_features['feature'].tolist()
                # 添加节点及其属性
                G.add_node(stock, 
                          value_type='high' if stock in self.high_value_stocks else 'low',
                          top_features=top_feature_names)
            else:
                # 如果没有特征重要性信息，简单添加节点
                G.add_node(stock, 
                          value_type='high' if stock in self.high_value_stocks else 'low')
        
        # 添加边
        for i, stock1 in enumerate(stock_list):
            for j in range(i+1, len(stock_list)):
                stock2 = stock_list[j]
                # 使用精确矩阵中的值作为边权重
                if abs(precision_matrix[i, j]) > 1e-5:  # 避免非常小的值
                    # 找出两只股票的共同重要特征
                    common_features = self._get_common_important_features(stock1, stock2)
                    
                    G.add_edge(stock1, stock2, 
                              weight=abs(precision_matrix[i, j]),
                              common_features=common_features)
        
        return G
    
    def _calculate_feature_similarity(self, stock1, stock2):
        """计算两只股票基于重要特征的相似度"""
        # 如果无法找到这些股票的数据，返回0
        if stock1 not in self.data.columns.get_level_values(0) or stock2 not in self.data.columns.get_level_values(0):
            return 0
        
        # 获取共同特征
        stock1_features = set(self.data[stock1].columns)
        stock2_features = set(self.data[stock2].columns)
        common_features = stock1_features.intersection(stock2_features)
        
        if not common_features:
            return 0  # 没有共同特征
        
        # 使用共同特征计算相关性
        similarity_scores = []
        for feature in common_features:
            # 获取两只股票的特征时间序列
            series1 = self.data[stock1][feature]
            series2 = self.data[stock2][feature]
            
            # 计算相关系数
            try:
                # 确保数据有效
                valid_data = ~(np.isnan(series1) | np.isnan(series2))
                if np.sum(valid_data) > 10:  # 至少需要10个有效数据点
                    corr = np.corrcoef(series1[valid_data], series2[valid_data])[0, 1]
                    
                    # 根据特征重要性加权相关系数
                    importance1 = self._get_feature_importance(stock1, feature)
                    importance2 = self._get_feature_importance(stock2, feature)
                    
                    # 使用两只股票特征重要性的平均值作为权重
                    weight = (importance1 + importance2) / 2
                    
                    # 加权相关系数
                    similarity_scores.append(abs(corr) * weight)
            except:
                continue
        
        # 返回加权平均相似度
        return np.mean(similarity_scores) if similarity_scores else 0
    
    def _get_feature_importance(self, stock, feature):
        """获取特定股票特定特征的重要性"""
        if stock in self.feature_importances:
            importance_df = self.feature_importances[stock]
            if feature in importance_df['feature'].values:
                return importance_df[importance_df['feature'] == feature]['importance'].values[0]
        
        # 如果找不到重要性信息，返回默认值1.0
        return 1.0
    
    def _get_common_important_features(self, stock1, stock2):
        """找出两只股票共同的重要特征"""
        # 如果没有重要性信息，返回空列表
        if stock1 not in self.feature_importances or stock2 not in self.feature_importances:
            return []
        
        # 获取每只股票的重要特征(取前5个)
        stock1_features = set(self.feature_importances[stock1].sort_values('importance', ascending=False).head(5)['feature'])
        stock2_features = set(self.feature_importances[stock2].sort_values('importance', ascending=False).head(5)['feature'])
        
        # 返回共同的重要特征
        return list(stock1_features.intersection(stock2_features))
    
    def build_cross_value_network(self, high_repr_stocks, low_repr_stocks):
        """构建高价值和低价值代表性股票之间的网络"""
        cross_network = nx.Graph(name="cross_value_network")
        
        # 添加所有代表性股票作为节点
        for stock in high_repr_stocks:
            cross_network.add_node(stock, value_type='high')
            if stock in self.feature_importances:
                top_features = self.feature_importances[stock].sort_values('importance', ascending=False).head(3)['feature'].tolist()
                cross_network.nodes[stock]['top_features'] = top_features
        
        for stock in low_repr_stocks:
            cross_network.add_node(stock, value_type='low')
            if stock in self.feature_importances:
                top_features = self.feature_importances[stock].sort_values('importance', ascending=False).head(3)['feature'].tolist()
                cross_network.nodes[stock]['top_features'] = top_features
        
        # 计算高价值和低价值股票之间的相似性并添加边
        for high_stock in high_repr_stocks:
            for low_stock in low_repr_stocks:
                similarity = self._calculate_feature_similarity(high_stock, low_stock)
                if similarity > 0.3:  # 设置一个阈值
                    common_features = self._get_common_important_features(high_stock, low_stock)
                    cross_network.add_edge(high_stock, low_stock, 
                                         weight=similarity,
                                         common_features=common_features)
        
        return cross_network