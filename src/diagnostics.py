"""
Diagnostics Module
Model validation and error analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class Diagnostics:
    def __init__(self):
        self.diagnostic_results = {}
        
    def calculate_prediction_errors(self, y_true, y_pred):
        """Calculate various error metrics"""
        errors = y_pred - y_true
        absolute_errors = np.abs(errors)
        
        diagnostics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R²': r2_score(y_true, y_pred),
            'Max Error': np.max(absolute_errors),
            'Min Error': np.min(absolute_errors),
            'Mean Error': np.mean(errors),
            'Std Dev Error': np.std(errors),
            'Median Error': np.median(errors),
            'Q1 Error': np.percentile(errors, 25),
            'Q3 Error': np.percentile(errors, 75),
        }
        
        return diagnostics
    
    def identify_problem_regions(self, y_true, y_pred, threshold_percentile=90):
        """Identify where model performs poorly"""
        errors = np.abs(y_pred - y_true)
        threshold = np.percentile(errors, threshold_percentile)
        
        problem_indices = np.where(errors > threshold)[0]
        problem_percentage = len(problem_indices) / len(errors) * 100
        
        logger.info(f"\nProblem Identification:")
        logger.info(f"  Errors above {threshold_percentile}th percentile: {len(problem_indices)}")
        logger.info(f"  Percentage of data: {problem_percentage:.2f}%")
        logger.info(f"  Error threshold: {threshold:.6f}")
        
        return problem_indices, threshold
    
    def residual_normality_test(self, residuals):
        """Test normality of residuals using Shapiro-Wilk test"""
        from scipy.stats import shapiro
        
        # Sample for large datasets (Shapiro-Wilk has 5000 sample limit)
        sample_size = min(5000, len(residuals))
        sample_residuals = np.random.choice(residuals, size=sample_size, replace=False)
        
        stat, p_value = shapiro(sample_residuals)
        
        logger.info(f"\nShapiro-Wilk Normality Test:")
        logger.info(f"  Test Statistic: {stat:.6f}")
        logger.info(f"  P-value: {p_value:.10f}")
        
        if p_value > 0.05:
            logger.info("  Result: Residuals appear normally distributed (p > 0.05)")
        else:
            logger.info("  Result: Residuals may not be normally distributed (p < 0.05)")
        
        return stat, p_value
    
    def residual_autocorrelation_test(self, residuals):
        """Test for autocorrelation using Durbin-Watson statistic"""
        from scipy.stats import norm
        
        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        
        logger.info(f"\nDurbin-Watson Autocorrelation Test:")
        logger.info(f"  DW Statistic: {dw:.6f}")
        logger.info(f"  (Value near 2 suggests no autocorrelation)")
        
        return dw
    
    def heteroscedasticity_test(self, residuals, y_pred):
        """Test for heteroscedasticity (non-constant variance)"""
        from scipy.stats import spearmanr
        
        # Spearman correlation between absolute residuals and predicted values
        corr, p_value = spearmanr(np.abs(residuals), y_pred)
        
        logger.info(f"\nHeteroscedasticity Test (Spearman):")
        logger.info(f"  Correlation: {corr:.6f}")
        logger.info(f"  P-value: {p_value:.10f}")
        
        if p_value > 0.05:
            logger.info("  Result: Constant variance appears likely (p > 0.05)")
        else:
            logger.info("  Result: Heteroscedasticity detected (p < 0.05)")
        
        return corr, p_value
    
    def model_error_distribution_by_magnitude(self, y_true, y_pred):
        """Analyze prediction errors by target magnitude"""
        errors = np.abs(y_pred - y_true)
        quartiles = np.quantile(y_true, [0.25, 0.5, 0.75])
        
        logger.info(f"\nError Distribution by Target Magnitude:")
        logger.info(f"  Bottom 25% (y < {quartiles[0]:.2f}): MAE = {errors[y_true <= quartiles[0]].mean():.6f}")
        logger.info(f"  25-50% ({quartiles[0]:.2f} <= y < {quartiles[1]:.2f}): MAE = {errors[(y_true > quartiles[0]) & (y_true <= quartiles[1])].mean():.6f}")
        logger.info(f"  50-75% ({quartiles[1]:.2f} <= y < {quartiles[2]:.2f}): MAE = {errors[(y_true > quartiles[1]) & (y_true <= quartiles[2])].mean():.6f}")
        logger.info(f"  Top 25% (y >= {quartiles[2]:.2f}): MAE = {errors[y_true >= quartiles[2]].mean():.6f}")
