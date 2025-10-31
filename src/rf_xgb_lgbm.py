"""
Tree-based and Ensemble Models with GPU Acceleration
Random Forest, XGBoost, LightGBM
Optimized for CUDA 13 and RTX GPUs (including RTX 3050)
FIXED: Proper GPU device handling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


class TreeBasedModels:
    def __init__(self, gpu_available=True):
        self.models = {}
        self.gpu_available = gpu_available
        
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=20):
        """Train Random Forest Regressor"""
        logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        self.models['RandomForest'] = model
        
        train_score = model.score(X_train, y_train)
        logger.info(f"Random Forest Train R²: {train_score:.6f}")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost with GPU acceleration (histogram method)"""
        logger.info("Training XGBoost with GPU acceleration...")
        
        # XGBoost parameters optimized for GPU and competition
        # Using histogram method which is GPU-compatible
        params = {
            'objective': 'reg:squarederror',
            'metric': 'mae',
            'tree_method': 'hist',  # GPU-compatible histogram method
            'gpu_id': 0,  # Use first GPU
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 0.5,
            'reg_lambda': 1,
            'min_child_weight': 1,
            'scale_pos_weight': 1,
            'seed': 42,
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
            evals_result = {}
        else:
            evals = [(dtrain, 'train')]
            evals_result = {}
        
        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=evals,
                evals_result=evals_result,
                early_stopping_rounds=50,
                verbose_eval=50
            )
            self.models['XGBoost'] = model
            logger.info("XGBoost training complete (GPU accelerated)")
        except Exception as e:
            logger.warning(f"GPU training failed, retrying with CPU: {e}")
            # Fallback to CPU
            params['tree_method'] = 'auto'
            params.pop('gpu_id', None)
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=50
            )
            self.models['XGBoost'] = model
            logger.info("XGBoost training complete (CPU fallback)")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM with GPU acceleration"""
        logger.info("Training LightGBM with GPU acceleration...")
        
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'device': 'gpu',
            'gpu_device_id': 0,
            'max_depth': 7,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 1,
            'min_child_samples': 20,
            'verbose': -1,
            'seed': 42,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(50)
                ]
            )
            self.models['LightGBM'] = model
            logger.info("LightGBM training complete (GPU accelerated)")
        except Exception as e:
            logger.warning(f"GPU training failed, retrying with CPU: {e}")
            # Fallback to CPU
            params['device'] = 'cpu'
            params.pop('gpu_device_id', None)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(50)
                ]
            )
            self.models['LightGBM'] = model
            logger.info("LightGBM training complete (CPU fallback)")
        
        return model
    
    def get_feature_importance(self, model_name):
        """Extract feature importance from tree models"""
        model = self.models.get(model_name)
        
        if model_name == 'RandomForest':
            return model.feature_importances_
        elif model_name == 'XGBoost':
            booster = model
            importance = booster.get_score(importance_type='weight')
            return importance
        elif model_name == 'LightGBM':
            return model.feature_importance(importance_type='split')
        
        return None