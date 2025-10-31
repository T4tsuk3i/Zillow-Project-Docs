"""
Feature Engineering Module for Zillow Prize
Advanced feature transformation and selection
GPU-Compatible
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self):
        self.feature_stats = {}
        
    def create_interaction_features(self, df, features_list=None):
        """Create interaction features from selected columns"""
        if features_list is None:
            features_list = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet']
        
        available_features = [f for f in features_list if f in df.columns]
        
        for i, f1 in enumerate(available_features):
            for f2 in available_features[i+1:]:
                df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
        
        logger.info(f"Created interaction features from {available_features}")
        return df
    
    def create_polynomial_features(self, df, features_list=None, degree=2):
        """Create polynomial features (limited to avoid explosion)"""
        if features_list is None:
            features_list = ['lotsizesquarefeet', 'taxvaluedollarcnt']
        
        available_features = [f for f in features_list if f in df.columns]
        
        for feat in available_features:
            for d in range(2, degree+1):
                df[f'{feat}_pow_{d}'] = df[feat] ** d
        
        logger.info(f"Created polynomial features up to degree {degree}")
        return df
    
    def create_aggregate_features(self, df):
        """Create aggregate features based on geography"""
        if 'fips' in df.columns:
            df['median_house_value_by_fips'] = df.groupby('fips')['taxvaluedollarcnt'].transform('median')
            df['mean_price_sqft_by_fips'] = df.groupby('fips')['price_per_sqft'].transform('mean')
        
        if 'regionidcity' in df.columns:
            df['median_beds_by_city'] = df.groupby('regionidcity')['bedroomcnt'].transform('median')
            df['median_baths_by_city'] = df.groupby('regionidcity')['bathroomcnt'].transform('median')
        
        logger.info("Created aggregate features")
        return df
    
    def detect_feature_importance(self, X, y):
        """Correlation-based feature importance"""
        from sklearn.preprocessing import StandardScaler
        
        X_scaled = StandardScaler().fit_transform(X)
        correlations = np.abs(np.corrcoef(X_scaled.T, y.values)[:-1, -1])
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'correlation': correlations
        }).sort_values('correlation', ascending=False)
        
        self.feature_stats['correlation'] = feature_importance
        logger.info("\nTop 20 Features by Correlation:")
        logger.info(feature_importance.head(20).to_string())
        
        return feature_importance
    
    def select_top_features(self, X, y, n_features=50):
        """Select top features by correlation"""
        importance = self.detect_feature_importance(X, y)
        top_features = importance.head(n_features)['feature'].tolist()
        
        logger.info(f"Selected top {n_features} features")
        return top_features
    
    def create_binned_features(self, df):
        """Create binned categorical features from continuous"""
        if 'property_age' in df.columns:
            df['age_bin'] = pd.qcut(df['property_age'], q=5, labels=False, duplicates='drop')
        
        if 'taxvaluedollarcnt' in df.columns:
            df['value_bin'] = pd.qcut(df['taxvaluedollarcnt'], q=5, labels=False, duplicates='drop')
        
        if 'calculatedfinishedsquarefeet' in df.columns:
            df['sqft_bin'] = pd.qcut(df['calculatedfinishedsquarefeet'], q=5, labels=False, duplicates='drop')
        
        logger.info("Created binned features")
        return df