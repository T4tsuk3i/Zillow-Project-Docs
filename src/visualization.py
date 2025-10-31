"""
Visualization Module - FIXED
Create diagnostic plots and feature importance visualizations
Handles both numeric and string columns properly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class Visualization:
    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        
    def plot_residuals(self, y_true, y_pred, model_name):
        """Create residual diagnostic plots"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Residual Diagnostics - {model_name}', fontsize=16, fontweight='bold')
        
        # Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Normal Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scale-Location plot
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
        axes[1, 1].set_xlabel('Fitted Values')
        axes[1, 1].set_ylabel('√|Standardized Residuals|')
        axes[1, 1].set_title('Scale-Location Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/residuals_{model_name}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Residual plot saved for {model_name}")
        plt.close()
    
    def plot_feature_importance(self, feature_importance, feature_names, model_name, top_n=20):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if isinstance(feature_importance, dict):
            # For XGBoost
            indices = sorted(range(len(feature_importance)), key=lambda i: list(feature_importance.values())[i], reverse=True)[:top_n]
            importance_values = [list(feature_importance.values())[i] for i in indices]
            feature_labels = [list(feature_importance.keys())[i] for i in indices]
        else:
            # For other models
            indices = np.argsort(feature_importance)[-top_n:][::-1]
            importance_values = feature_importance[indices]
            feature_labels = [feature_names[i] for i in indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_values)))
        ax.barh(range(len(importance_values)), importance_values, color=colors)
        ax.set_yticks(range(len(importance_values)))
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved for {model_name}")
        plt.close()
    
    def plot_actual_vs_predicted(self, y_true, y_pred, model_name):
        """Plot actual vs predicted"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Actual vs Predicted - {model_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/actual_vs_predicted_{model_name}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Actual vs Predicted plot saved for {model_name}")
        plt.close()
    
    def plot_correlation_heatmap(self, df, top_n=15):
        """Plot correlation heatmap - FIXED to handle non-numeric columns"""
        # Select ONLY numeric columns to avoid date string conversion errors
        numeric_df = df.select_dtypes(include=[np.number])
        
        if 'logerror' not in numeric_df.columns:
            logger.warning("logerror not found in numeric columns")
            return
        
        corr = numeric_df.corr()
        
        # Get top correlations with target
        target_corr = corr['logerror'].abs().sort_values(ascending=False)[1:top_n+1]
        
        # Create subset for heatmap
        correlation_subset = corr.loc[target_corr.index.tolist() + ['logerror'], 
                                       target_corr.index.tolist() + ['logerror']]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                    cbar_kws={'label': 'Correlation'})
        plt.title(f'Correlation Matrix - Top {top_n} Features with logerror', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info("Correlation heatmap saved")
        plt.close()
    
    def plot_target_distribution(self, y, target_name='logerror'):
        """Plot target variable distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel(target_name)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of {target_name}')
        axes[0].grid(True, alpha=0.3)
        
        stats.probplot(y, dist="norm", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot of {target_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Target distribution plot saved")
        plt.close()