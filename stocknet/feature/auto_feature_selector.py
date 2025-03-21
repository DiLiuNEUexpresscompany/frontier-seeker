import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

class AutoFeatureSelector:
    """
    自动特征选择器
    
    结合多种特征选择策略，自动为每只股票选择最佳特征集。
    可以根据预测性能自动选择最佳特征子集。
    
    属性:
        max_features (int): 每只股票保留的最大特征数
        n_jobs (int): 并行计算的作业数
        cv_folds (int): 交叉验证折数
    """
    
    def __init__(self, max_features: int = 5, n_jobs: int = 4, cv_folds: int = 5):
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.logger = logging.getLogger('AutoFeatureSelector')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.selected_features = {}
        self.feature_importances = {}
    
    def select_features(self, data: pd.DataFrame, future_periods: List[int] = [1, 5, 10], target_col: str = 'close', method: str = 'auto') -> pd.DataFrame:
        """
        自动选择特征
        
        参数:
            data (pd.DataFrame): 股票数据，MultiIndex列格式
            future_periods (List[int]): 未来收益期限列表
            target_col (str): 目标列名
            method (str): 特征选择方法 ('auto', 'forward', 'backward', 'recursive')
            
        返回:
            pd.DataFrame: 选择特征后的数据
        """
        if not isinstance(data.columns, pd.MultiIndex):
            self.logger.error("输入数据列不是MultiIndex格式")
            return data
        
        stocks = data.columns.get_level_values(0).unique()
        self.logger.info(f"为 {len(stocks)} 只股票自动选择特征")
        all_selected_cols = []
        
        if len(stocks) > 20 and self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {}
                for stock in stocks:
                    futures[stock] = executor.submit(self._select_stock_features, data[stock].copy(), future_periods, target_col, method)
                for stock, future in tqdm(futures.items(), desc="自动特征选择"):
                    try:
                        selected, importance = future.result()
                        self.selected_features[stock] = selected
                        self.feature_importances[stock] = importance
                        for feature in selected:
                            all_selected_cols.append((stock, feature))
                    except Exception as e:
                        self.logger.error(f"处理股票 {stock} 自动特征选择出错: {e}")
                        self.selected_features[stock] = data[stock].columns.tolist()
                        for feature in data[stock].columns:
                            all_selected_cols.append((stock, feature))
        else:
            for stock in tqdm(stocks, desc="自动特征选择"):
                try:
                    stock_data = data[stock].copy()
                    selected, importance = self._select_stock_features(stock_data, future_periods, target_col, method)
                    self.selected_features[stock] = selected
                    self.feature_importances[stock] = importance
                    for feature in selected:
                        all_selected_cols.append((stock, feature))
                except Exception as e:
                    self.logger.error(f"处理股票 {stock} 自动特征选择出错: {e}")
                    self.selected_features[stock] = data[stock].columns.tolist()
                    for feature in data[stock].columns:
                        all_selected_cols.append((stock, feature))
        
        selected_data = data[all_selected_cols]
        self.logger.info(f"特征选择完成，从 {data.shape[1]} 个特征减少到 {selected_data.shape[1]} 个")
        return selected_data

    def _select_stock_features(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str, method: str) -> Tuple[List[str], pd.DataFrame]:
        if stock_data.empty:
            return [], pd.DataFrame()
        if target_col not in stock_data.columns:
            target_col = stock_data.columns[0]
        if len(stock_data.columns) <= self.max_features:
            importance = pd.DataFrame({'feature': stock_data.columns, 'importance': 1.0})
            return stock_data.columns.tolist(), importance
        if method == 'auto':
            return self._select_best_method(stock_data, future_periods, target_col)
        elif method == 'forward':
            return self._forward_selection(stock_data, future_periods, target_col)
        elif method == 'backward':
            return self._backward_elimination(stock_data, future_periods, target_col)
        elif method == 'recursive':
            return self._recursive_feature_elimination(stock_data, future_periods, target_col)
        else:
            self.logger.warning(f"未知特征选择方法: {method}，使用auto方法")
            return self._select_best_method(stock_data, future_periods, target_col)

    def _select_best_method(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str) -> Tuple[List[str], pd.DataFrame]:
        mi_features, mi_importance = self._select_by_mutual_info_with_future(stock_data, future_periods, target_col)
        corr_features, corr_importance = self._select_by_correlation_with_future(stock_data, future_periods, target_col)
        pca_features, pca_importance = self._select_by_pca_importance(stock_data, target_col)
        all_features = list(set(mi_features + corr_features + pca_features))
        if len(all_features) > self.max_features:
            feature_counts = {}
            for feature in all_features:
                count = 0
                if feature in mi_features:
                    count += 1
                if feature in corr_features:
                    count += 1
                if feature in pca_features:
                    count += 1
                feature_counts[feature] = count
            sorted_features = sorted(feature_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
            selected_features = [f[0] for f in sorted_features[:self.max_features]]
            if target_col not in selected_features:
                selected_features[-1] = target_col
        else:
            selected_features = all_features
            if target_col not in selected_features:
                selected_features.append(target_col)
                if len(selected_features) > self.max_features:
                    for i in range(len(selected_features)-1, -1, -1):
                        if selected_features[i] != target_col:
                            selected_features.pop(i)
                            break
        importance = pd.DataFrame({'feature': selected_features})
        importance['mi_importance'] = importance['feature'].map(dict(zip(mi_importance['feature'], mi_importance['importance']))).fillna(0)
        importance['corr_importance'] = importance['feature'].map(dict(zip(corr_importance['feature'], corr_importance['importance']))).fillna(0)
        importance['pca_importance'] = importance['feature'].map(dict(zip(pca_importance['feature'], pca_importance['importance']))).fillna(0)
        importance['importance'] = (importance['mi_importance'] + importance['corr_importance'] + importance['pca_importance']) / 3
        return selected_features, importance

    def _select_by_mutual_info_with_future(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str) -> Tuple[List[str], pd.DataFrame]:
        close_prices = stock_data[target_col].values
        future_returns = {}
        for period in future_periods:
            returns = np.zeros_like(close_prices)
            returns[:-period] = close_prices[period:] / close_prices[:-period] - 1
            future_returns[period] = returns
        features = stock_data.copy()
        mi_scores = {}
        for feature in features.columns:
            feature_values = features[feature].values
            period_scores = []
            for period, returns in future_returns.items():
                valid_idx = ~(np.isnan(feature_values) | np.isnan(returns))
                if np.sum(valid_idx) < 30:
                    period_scores.append(0)
                    continue
                try:
                    mi = mutual_info_regression(feature_values[valid_idx].reshape(-1, 1), returns[valid_idx])[0]
                    period_scores.append(mi)
                except Exception:
                    period_scores.append(0)
            mi_scores[feature] = np.mean(period_scores)
        importance_df = pd.DataFrame({'feature': list(mi_scores.keys()), 'importance': list(mi_scores.values())})
        top_features = importance_df.sort_values('importance', ascending=False)
        if target_col in top_features['feature'].values:
            selected = top_features.head(self.max_features)['feature'].tolist()
        else:
            selected = [target_col] + top_features.head(self.max_features - 1)['feature'].tolist()
        return selected, importance_df

    def _select_by_correlation_with_future(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str) -> Tuple[List[str], pd.DataFrame]:
        df = stock_data.copy()
        for period in future_periods:
            df[f'future_return_{period}'] = df[target_col].pct_change(period).shift(-period)
        future_cols = [f'future_return_{period}' for period in future_periods]
        corr_matrix = df.corr().abs()
        avg_future_corr = {}
        for feature in stock_data.columns:
            feature_corrs = []
            for future_col in future_cols:
                if future_col in corr_matrix.columns:
                    feature_corrs.append(corr_matrix.loc[feature, future_col])
            avg_future_corr[feature] = np.mean(feature_corrs) if feature_corrs else 0
        importance_df = pd.DataFrame({'feature': list(avg_future_corr.keys()), 'importance': list(avg_future_corr.values())})
        top_features = importance_df.sort_values('importance', ascending=False)
        if target_col in top_features['feature'].values:
            selected = top_features.head(self.max_features)['feature'].tolist()
        else:
            selected = [target_col] + top_features.head(self.max_features - 1)['feature'].tolist()
        return selected, importance_df

    def _select_by_pca_importance(self, stock_data: pd.DataFrame, target_col: str) -> Tuple[List[str], pd.DataFrame]:
        #X = stock_data.fillna(method='ffill').fillna(method='bfill').values
        # 修复后的代码:
        X = stock_data.ffill().bfill().values
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(len(stock_data.columns), 10))
        pca.fit(X)
        components = np.abs(pca.components_)
        feature_importance = np.sum(components, axis=0)
        importance_df = pd.DataFrame({'feature': stock_data.columns, 'importance': feature_importance})
        top_features = importance_df.sort_values('importance', ascending=False)
        if target_col in top_features.head(self.max_features)['feature'].values:
            selected = top_features.head(self.max_features)['feature'].tolist()
        else:
            selected = [target_col] + top_features.head(self.max_features - 1)['feature'].tolist()
        return selected, importance_df

    def _forward_selection(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str) -> Tuple[List[str], pd.DataFrame]:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        df = stock_data.copy().fillna(method='ffill').fillna(method='bfill')
        y = np.zeros(len(df))
        for period in future_periods:
            future_return = df[target_col].pct_change(period).shift(-period)
            y += future_return.fillna(0).values
        y /= len(future_periods)
        valid_idx = ~np.isnan(y)
        X_full = df.values[valid_idx, :]
        y = y[valid_idx]
        feature_names = df.columns.tolist()
        X_full = StandardScaler().fit_transform(X_full)
        n_samples, n_features = X_full.shape
        selected_features_idx = []
        selected_features_names = []
        current_score = float('-inf')
        feature_scores = {}
        for _ in range(min(self.max_features, n_features)):
            best_score = float('-inf')
            best_feature = -1
            for feature_idx in range(n_features):
                if feature_idx in selected_features_idx:
                    continue
                current_features = selected_features_idx + [feature_idx]
                X = X_full[:, current_features]
                cv_scores = []
                n_splits = min(self.cv_folds, 5)
                split_size = n_samples // n_splits
                for i in range(n_splits):
                    test_start = i * split_size
                    test_end = (i + 1) * split_size if i < n_splits - 1 else n_samples
                    test_idx = np.arange(test_start, test_end)
                    train_idx = np.array([j for j in range(n_samples) if j not in test_idx])
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    score = -mse
                    cv_scores.append(score)
                avg_score = np.mean(cv_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_feature = feature_idx
            if best_score <= current_score:
                break
            current_score = best_score
            selected_features_idx.append(best_feature)
            feature_name = feature_names[best_feature]
            selected_features_names.append(feature_name)
            feature_scores[feature_name] = best_score
        if target_col not in selected_features_names:
            selected_features_names.insert(0, target_col)
            if len(selected_features_names) > self.max_features:
                for i in range(len(selected_features_names)-1, -1, -1):
                    if selected_features_names[i] != target_col:
                        selected_features_names.pop(i)
                        break
        importance_df = pd.DataFrame({'feature': list(feature_scores.keys()), 'importance': list(feature_scores.values())})
        if target_col not in importance_df['feature'].values:
            importance_df = pd.concat([pd.DataFrame({'feature': [target_col], 'importance': [max(feature_scores.values()) if feature_scores else 1.0]}), importance_df]).reset_index(drop=True)
        if len(selected_features_names) > self.max_features:
            selected_features_names = selected_features_names[:self.max_features]
        return selected_features_names, importance_df

    def _backward_elimination(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str) -> Tuple[List[str], pd.DataFrame]:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        df = stock_data.copy().fillna(method='ffill').fillna(method='bfill')
        y = np.zeros(len(df))
        for period in future_periods:
            future_return = df[target_col].pct_change(period).shift(-period)
            y += future_return.fillna(0).values
        y /= len(future_periods)
        valid_idx = ~np.isnan(y)
        X_full = df.values[valid_idx, :]
        y = y[valid_idx]
        feature_names = df.columns.tolist()
        X_full = StandardScaler().fit_transform(X_full)
        n_samples, n_features = X_full.shape
        remaining_features_idx = list(range(n_features))
        remaining_feature_names = feature_names.copy()
        current_score = float('-inf')
        feature_scores = {}
        X = X_full.copy()
        cv_scores = []
        n_splits = min(self.cv_folds, 5)
        split_size = n_samples // n_splits
        for i in range(n_splits):
            test_start = i * split_size
            test_end = (i + 1) * split_size if i < n_splits - 1 else n_samples
            test_idx = np.arange(test_start, test_end)
            train_idx = np.array([j for j in range(n_samples) if j not in test_idx])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            score = -mse
            cv_scores.append(score)
        current_score = np.mean(cv_scores)
        while len(remaining_features_idx) > self.max_features:
            worst_score = float('inf')
            worst_feature = -1
            for i, feature_idx in enumerate(remaining_features_idx):
                if remaining_feature_names[i] == target_col:
                    continue
                current_features = remaining_features_idx.copy()
                current_features.remove(feature_idx)
                X = X_full[:, current_features]
                cv_scores = []
                for j in range(n_splits):
                    test_start = j * split_size
                    test_end = (j + 1) * split_size if j < n_splits - 1 else n_samples
                    test_idx = np.arange(test_start, test_end)
                    train_idx = np.array([k for k in range(n_samples) if k not in test_idx])
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    score = -mse
                    cv_scores.append(score)
                avg_score = np.mean(cv_scores)
                if avg_score < worst_score:
                    worst_score = avg_score
                    worst_feature = i
            if worst_feature == -1 or remaining_feature_names[worst_feature] == target_col:
                break
            removed_feature_name = remaining_feature_names[worst_feature]
            feature_scores[removed_feature_name] = worst_score
            removed_feature_idx = remaining_features_idx[worst_feature]
            remaining_features_idx.remove(removed_feature_idx)
            remaining_feature_names.pop(worst_feature)
        for feature in remaining_feature_names:
            if feature not in feature_scores:
                feature_scores[feature] = current_score * 1.1
        importance_df = pd.DataFrame({'feature': list(feature_scores.keys()), 'importance': list(feature_scores.values())})
        importance_df = importance_df.sort_values('importance', ascending=False)
        return remaining_feature_names, importance_df

    def _recursive_feature_elimination(self, stock_data: pd.DataFrame, future_periods: List[int], target_col: str) -> Tuple[List[str], pd.DataFrame]:
        from sklearn.feature_selection import RFECV
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import TimeSeriesSplit
        df = stock_data.copy().fillna(method='ffill').fillna(method='bfill')
        y = np.zeros(len(df))
        for period in future_periods:
            future_return = df[target_col].pct_change(period).shift(-period)
            y += future_return.fillna(0).values
        y /= len(future_periods)
        valid_idx = ~np.isnan(y)
        X = df.values[valid_idx, :]
        y = y[valid_idx]
        feature_names = df.columns.tolist()
        X = StandardScaler().fit_transform(X)
        try:
            tscv = TimeSeriesSplit(n_splits=min(self.cv_folds, 5))
            model = Ridge(alpha=1.0)
            selector = RFECV(model, step=1, min_features_to_select=1, cv=tscv, scoring='neg_mean_squared_error')
            selector = selector.fit(X, y)
            selected_idx = np.where(selector.support_)[0]
            selected_features = [feature_names[i] for i in selected_idx]
            if target_col not in selected_features:
                selected_features.insert(0, target_col)
            if len(selected_features) > self.max_features:
                if target_col in selected_features:
                    target_idx = selected_features.index(target_col)
                    selected_features.pop(target_idx)
                    ranking = selector.ranking_
                    top_features = [feature_names[i] for i in np.argsort(ranking)[:self.max_features-1]]
                    selected_features = [target_col] + top_features
                else:
                    ranking = selector.ranking_
                    selected_features = [feature_names[i] for i in np.argsort(ranking)[:self.max_features]]
            importance = 1 / selector.ranking_
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
            return selected_features, importance_df
        except Exception as e:
            self.logger.warning(f"递归特征消除失败: {e}，使用互信息方法代替")
            return self._select_by_mutual_info_with_future(stock_data, future_periods, target_col)

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """获取每只股票的特征重要性"""
        return self.feature_importances
