"""
Multi-Output Model - Ensemble Stacking
Combines predictions from multiple models using a meta-learner
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
import logging

logger = logging.getLogger(__name__)


class MultiOutputModel:
    """Stacked Ensemble using multiple base models"""
    
    def __init__(self, base_models=None):
        self.base_models = base_models or {}
        self.meta_model = None
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
    def prepare_meta_features(self, X, y, base_models_list):
        """Generate meta-features using cross-validation"""
        logger.info("Preparing meta-features for stacking...")
        
        meta_features = np.zeros((X.shape[0], len(base_models_list)))
        
        for i, (model_name, model) in enumerate(base_models_list):
            logger.info(f"  Generating meta-features for {model_name}")
            
            # Get cross-validated predictions
            if hasattr(model, 'predict'):
                meta_features[:, i] = cross_val_predict(model, X, y, cv=self.cv)
            else:
                # For already trained models (XGBoost, LightGBM)
                meta_features[:, i] = model.predict(X)
        
        logger.info(f"Meta-features shape: {meta_features.shape}")
        return meta_features
    
    def train_meta_model(self, meta_features, y):
        """Train the meta-learner (typically Ridge regression)"""
        logger.info("Training meta-learner...")
        
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y)
        
        logger.info("Meta-model training complete")
        return self.meta_model
    
    def predict(self, meta_features):
        """Make predictions using the meta-model"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained")
        
        return self.meta_model.predict(meta_features)
