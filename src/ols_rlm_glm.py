"""
Linear Models Module (OLS, RLM, GLM, Ridge, Lasso, ElasticNet)
Statsmodels and Scikit-learn implementations
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)


class LinearModels:
    def __init__(self):
        self.models = {}
        
    def train_ols(self, X_train, y_train):
        """Ordinary Least Squares"""
        logger.info("Training OLS Model...")
        X_train_sm = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_sm).fit()
        self.models['OLS'] = model
        logger.info(f"OLS R-squared: {model.rsquared:.6f}")
        return model
    
    def train_rlm(self, X_train, y_train):
        """Robust Linear Model (RLM)"""
        logger.info("Training RLM Model...")
        X_train_sm = sm.add_constant(X_train)
        model = sm.RLM(y_train, X_train_sm, M=sm.robust.norms.HuberT()).fit()
        self.models['RLM'] = model
        logger.info(f"RLM Training Complete")
        return model
    
    def train_glm(self, X_train, y_train):
        """Generalized Linear Model"""
        logger.info("Training GLM Model...")
        X_train_sm = sm.add_constant(X_train)
        model = sm.GLM(y_train, X_train_sm, family=sm.families.Gaussian()).fit()
        self.models['GLM'] = model
        logger.info(f"GLM Deviance: {model.deviance:.6f}")
        return model
    
    def train_ridge(self, X_train, y_train, alpha=1.0, cv=5):
        """Ridge Regression"""
        logger.info(f"Training Ridge Regression (alpha={alpha})...")
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['Ridge'] = model
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        logger.info(f"Ridge CV R² Mean: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")
        return model
    
    def train_lasso(self, X_train, y_train, alpha=0.01, cv=5):
        """Lasso Regression"""
        logger.info(f"Training Lasso Regression (alpha={alpha})...")
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_train, y_train)
        self.models['Lasso'] = model
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        logger.info(f"Lasso CV R² Mean: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")
        logger.info(f"Non-zero coefficients: {np.sum(model.coef_ != 0)}/{len(model.coef_)}")
        return model
    
    def train_elasticnet(self, X_train, y_train, alpha=0.01, l1_ratio=0.5, cv=5):
        """ElasticNet Regression"""
        logger.info(f"Training ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})...")
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        model.fit(X_train, y_train)
        self.models['ElasticNet'] = model
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
        logger.info(f"ElasticNet CV R² Mean: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")
        return model
    
    def get_coefficients(self, model_name):
        """Extract and return coefficients"""
        model = self.models.get(model_name)
        if model is None:
            logger.warning(f"Model {model_name} not found")
            return None
        
        if hasattr(model, 'params'):  # Statsmodels
            return model.params
        elif hasattr(model, 'coef_'):  # Scikit-learn
            return model.coef_
        return None
