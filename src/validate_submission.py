"""
Submission CSV Validator
Tests submission.csv predictions against actual logerror values
Calculates MAE, RMSE, and R² metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def validate_submission():
    """Validate submission CSV and calculate metrics"""
    
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION CSV VALIDATOR")
    logger.info("="*80 + "\n")
    
    data_dir = Path('data')
    output_dir = Path('outputs')
    
    # ========== LOAD SUBMISSION ==========
    logger.info("Step 1: Loading submission.csv...")
    
    submission_file = output_dir / 'submission2.csv'
    if not submission_file.exists():
        logger.error(f"  ✗ File not found: {submission_file}")
        logger.error("  Please run: python generate_submission_simple.py")
        return
    
    try:
        submission = pd.read_csv(submission_file)
        logger.info(f"  ✓ Submission loaded: {submission.shape}")
        logger.info(f"  ✓ Columns: {submission.columns.tolist()}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load submission: {e}")
        return
    
    # ========== LOAD TRAINING DATA ==========
    logger.info("\nStep 2: Loading training data...")
    
    try:
        train_2016 = pd.read_csv(data_dir / 'train_2016_v2.csv')
        logger.info(f"  ✓ Train 2016: {train_2016.shape}")
    except Exception as e:
        logger.warning(f"  ✗ Failed to load train 2016: {e}")
        train_2016 = None
    
    try:
        train_2017 = pd.read_csv(data_dir / 'train_2017.csv')
        logger.info(f"  ✓ Train 2017: {train_2017.shape}")
    except Exception as e:
        logger.warning(f"  ✗ Failed to load train 2017: {e}")
        train_2017 = None
    
    # Combine training data
    if train_2016 is not None and train_2017 is not None:
        train_data = pd.concat([train_2016, train_2017], ignore_index=True)
    elif train_2016 is not None:
        train_data = train_2016
    elif train_2017 is not None:
        train_data = train_2017
    else:
        logger.error("  ✗ No training data loaded!")
        return
    
    logger.info(f"  ✓ Combined training data: {train_data.shape}")
    logger.info(f"  ✓ Columns: {train_data.columns.tolist()}")
    
    # ========== MERGE AND EXTRACT ==========
    logger.info("\nStep 3: Merging submission with actual values...")
    
    # Rename ParcelId to parcelid for merge (if needed)
    submission_merged = submission.copy()
    if 'ParcelId' in submission_merged.columns:
        submission_merged['parcelid'] = submission_merged['ParcelId']
    
    train_data['parcelid'] = train_data['parcelid'].astype(int)
    submission_merged['parcelid'] = submission_merged['parcelid'].astype(int)
    
    # Get predictions (use first month as representative)
    if '201610' in submission_merged.columns:
        submission_merged['prediction'] = submission_merged['201610']
    else:
        logger.error("  ✗ No month columns found in submission!")
        return
    
    # Merge
    merged = train_data.merge(submission_merged[['parcelid', 'prediction']], on='parcelid', how='inner')
    
    logger.info(f"  ✓ Merged data: {merged.shape}")
    logger.info(f"  ✓ Actual logerror samples: {len(merged)}")
    
    if len(merged) == 0:
        logger.error("  ✗ No matching ParcelIds between submission and training data!")
        return
    
    # ========== EXTRACT PREDICTIONS AND ACTUAL ==========
    logger.info("\nStep 4: Extracting predictions and actual values...")
    
    y_actual = merged['logerror'].values
    y_pred = merged['prediction'].values
    
    logger.info(f"  ✓ Actual values: {len(y_actual)}")
    logger.info(f"  ✓ Predictions: {len(y_pred)}")
    logger.info(f"  ✓ Actual logerror range: [{y_actual.min():.6f}, {y_actual.max():.6f}]")
    logger.info(f"  ✓ Predicted logerror range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    
    # ========== CALCULATE METRICS ==========
    logger.info("\nStep 5: Calculating metrics...")
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-10))) * 100
    median_ae = np.median(np.abs(y_actual - y_pred))
    
    logger.info(f"\n  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.6f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  Median AE: {median_ae:.6f}")
    
    # ========== COMPARISON WITH BASELINE ==========
    logger.info("\nStep 6: Baseline comparison...")
    
    # Baseline: predict mean
    y_mean = np.full_like(y_actual, y_actual.mean())
    mae_baseline = mean_absolute_error(y_actual, y_mean)
    rmse_baseline = np.sqrt(mean_squared_error(y_actual, y_mean))
    r2_baseline = r2_score(y_actual, y_mean)
    
    logger.info(f"\n  Baseline (predicting mean):")
    logger.info(f"    MAE:  {mae_baseline:.6f}")
    logger.info(f"    RMSE: {rmse_baseline:.6f}")
    logger.info(f"    R²:   {r2_baseline:.6f}")
    
    logger.info(f"\n  Your Model:")
    logger.info(f"    MAE:  {mae:.6f} (baseline: {mae_baseline:.6f}, better by {(1 - mae/mae_baseline)*100:.1f}%)")
    logger.info(f"    RMSE: {rmse:.6f} (baseline: {rmse_baseline:.6f}, better by {(1 - rmse/rmse_baseline)*100:.1f}%)")
    logger.info(f"    R²:   {r2:.6f} (baseline: {r2_baseline:.6f}, improvement: {r2 - r2_baseline:.6f})")
    
    # ========== ERROR DISTRIBUTION ==========
    logger.info("\nStep 7: Error distribution analysis...")
    
    errors = np.abs(y_actual - y_pred)
    
    logger.info(f"\n  Error statistics:")
    logger.info(f"    Min error:    {errors.min():.6f}")
    logger.info(f"    Max error:    {errors.max():.6f}")
    logger.info(f"    Mean error:   {errors.mean():.6f}")
    logger.info(f"    Std error:    {errors.std():.6f}")
    logger.info(f"    25th percentile: {np.percentile(errors, 25):.6f}")
    logger.info(f"    50th percentile: {np.percentile(errors, 50):.6f}")
    logger.info(f"    75th percentile: {np.percentile(errors, 75):.6f}")
    logger.info(f"    95th percentile: {np.percentile(errors, 95):.6f}")
    
    # ========== PERFORMANCE BY MAGNITUDE ==========
    logger.info("\nStep 8: Performance by actual value magnitude...")
    
    percentiles = [25, 50, 75]
    for perc in percentiles:
        threshold = np.percentile(np.abs(y_actual), perc)
        mask = np.abs(y_actual) <= threshold
        
        mae_seg = mean_absolute_error(y_actual[mask], y_pred[mask])
        r2_seg = r2_score(y_actual[mask], y_pred[mask])
        
        logger.info(f"\n  Properties with |logerror| <= {threshold:.6f} ({perc}th percentile):")
        logger.info(f"    Samples: {mask.sum()}")
        logger.info(f"    MAE:  {mae_seg:.6f}")
        logger.info(f"    R²:   {r2_seg:.6f}")
    
    # ========== SAVE RESULTS ==========
    logger.info("\nStep 9: Saving validation results...")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R²', 'MAPE (%)', 'Median AE'],
        'Value': [mae, rmse, r2, mape, median_ae],
        'Baseline': [mae_baseline, rmse_baseline, r2_baseline, 
                     np.mean(np.abs((y_actual - y_mean) / (np.abs(y_actual) + 1e-10))) * 100,
                     np.median(np.abs(y_actual - y_mean))]
    })
    
    results_file = output_dir / 'submission_validation.csv'
    results_df.to_csv(results_file, index=False)
    logger.info(f"  ✓ Validation results saved to: {results_file}")
    
    # Save detailed predictions
    detailed_df = merged[['parcelid', 'logerror', 'prediction']].copy()
    detailed_df['error'] = np.abs(detailed_df['logerror'] - detailed_df['prediction'])
    detailed_df.to_csv(output_dir / 'submission_detailed_predictions.csv', index=False)
    logger.info(f"  ✓ Detailed predictions saved to: submission_detailed_predictions.csv")
    
    # ========== SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\nSubmission File: {submission_file}")
    logger.info(f"Training Samples: {len(merged)}")
    logger.info(f"\nMetrics:")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.6f}")
    
    if mae < 0.07:
        logger.info(f"\n  ✓ MAE is EXCELLENT (< 0.07)")
    elif mae < 0.08:
        logger.info(f"\n  ✓ MAE is VERY GOOD (< 0.08)")
    elif mae < 0.10:
        logger.info(f"\n  ✓ MAE is GOOD (< 0.10)")
    else:
        logger.info(f"\n  ⚠ MAE could be improved")
    
    if r2 > 0.3:
        logger.info(f"  ✓ R² is EXCELLENT (> 0.30)")
    elif r2 > 0.2:
        logger.info(f"  ✓ R² is VERY GOOD (> 0.20)")
    elif r2 > 0.1:
        logger.info(f"  ✓ R² is GOOD (> 0.10)")
    else:
        logger.info(f"  ⚠ R² could be improved")
    
    logger.info(f"\n" + "="*80 + "\n")
    
    return results_df, detailed_df


if __name__ == "__main__":
    validate_submission()