"""
Evaluation Module - FIXED
Calculate MAE, RMSE, R² and create comparison tables
GPU-Compatible
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  MAE:  {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R²:   {r2:.6f}")
        
        return metrics
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate predictions"""
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        self.results[model_name] = metrics
        
        return y_pred, metrics
    
    def create_comparison_table(self):
        """Create comparison table of all models"""
        if not self.results:
            logger.warning("No results to compare")
            return None
        
        comparison_df = pd.DataFrame(list(self.results.values()))
        comparison_df = comparison_df.sort_values('MAE')
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info(comparison_df.to_string(index=False))
        logger.info("="*80)
        
        return comparison_df
    
    def save_results(self, filepath='outputs/model_results.csv'):
        """Save results to CSV"""
        comparison_df = self.create_comparison_table()
        if comparison_df is not None:
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Results saved to {filepath}")
        
        return comparison_df