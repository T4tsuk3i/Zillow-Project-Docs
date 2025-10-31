"""
Simple Kaggle Submission Generator
Generates submission.csv from trained models
Run this AFTER main.py completes
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def generate_submission():
    """Generate Kaggle submission CSV"""
    
    logger.info("\n" + "="*80)
    logger.info("KAGGLE SUBMISSION GENERATOR")
    logger.info("="*80 + "\n")
    
    data_dir = Path('data')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # ========== LOAD DATA ==========
    logger.info("Step 1: Loading test properties...")
    
    try:
        properties_2016 = pd.read_csv(data_dir / 'properties_2016.csv', low_memory=False)
        logger.info(f"  ✓ 2016 properties: {properties_2016.shape}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load 2016: {e}")
        properties_2016 = None
    
    try:
        properties_2017 = pd.read_csv(data_dir / 'properties_2017.csv', low_memory=False)
        logger.info(f"  ✓ 2017 properties: {properties_2017.shape}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load 2017: {e}")
        properties_2017 = None
    
    # Combine
    if properties_2016 is not None and properties_2017 is not None:
        properties = pd.concat([properties_2016, properties_2017], ignore_index=True)
    elif properties_2016 is not None:
        properties = properties_2016
    elif properties_2017 is not None:
        properties = properties_2017
    else:
        logger.error("ERROR: No properties loaded!")
        return
    
    # Get unique properties
    properties = properties.drop_duplicates(subset=['parcelid'], keep='first')
    parcel_ids = properties['parcelid'].values
    logger.info(f"  ✓ Total unique properties: {len(parcel_ids):,}")
    
    # ========== PREPARE FEATURES ==========
    logger.info("\nStep 2: Preparing features...")
    
    # Select numeric columns only
    numeric_cols = properties.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target-like columns
    drop_cols = {'logerror', 'parcelid', 'transactiondate'}
    feature_cols = [col for col in numeric_cols if col not in drop_cols]
    
    logger.info(f"  ✓ Features available: {len(feature_cols)}")
    
    # Fill missing values with median
    X_all = properties[feature_cols].copy()
    for col in feature_cols:
        if X_all[col].isnull().sum() > 0:
            X_all[col] = X_all[col].fillna(X_all[col].median())
    
    logger.info(f"  ✓ Features shape: {X_all.shape}")
    logger.info(f"  ✓ Missing values: {X_all.isnull().sum().sum()}")
    
    # ========== LOAD TRAINING DATA ==========
    logger.info("\nStep 3: Loading training data for model training...")
    
    try:
        train_2016 = pd.read_csv(data_dir / 'train_2016_v2.csv')
        logger.info(f"  ✓ Train 2016: {train_2016.shape}")
    except:
        train_2016 = None
    
    try:
        train_2017 = pd.read_csv(data_dir / 'train_2017.csv')
        logger.info(f"  ✓ Train 2017: {train_2017.shape}")
    except:
        train_2017 = None
    
    # Merge with properties
    if train_2016 is not None:
        train_2016 = train_2016.merge(properties_2016, on='parcelid', how='left')
    
    if train_2017 is not None:
        train_2017 = train_2017.merge(properties_2017, on='parcelid', how='left')
    
    # Combine training data
    if train_2016 is not None and train_2017 is not None:
        train_data = pd.concat([train_2016, train_2017], ignore_index=True)
    elif train_2016 is not None:
        train_data = train_2016
    else:
        train_data = train_2017
    
    logger.info(f"  ✓ Combined training data: {train_data.shape}")
    
    # ========== PREPARE TRAINING DATA ==========
    logger.info("\nStep 4: Preparing training data...")
    
    X_train = train_data[feature_cols].copy()
    y_train = train_data['logerror'].copy()
    
    # Fill missing values
    for col in feature_cols:
        if X_train[col].isnull().sum() > 0:
            X_train[col] = X_train[col].fillna(X_train[col].median())
    
    # Remove any NaN rows
    valid_idx = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]
    
    logger.info(f"  ✓ Training data: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  ✓ Missing in training: {X_train.isnull().sum().sum()}")
    
    # ========== TRAIN LIGHTGBM ==========
    logger.info("\nStep 5: Training LightGBM on full training data...")
    
    try:
        params_lgb = {
            'objective': 'regression',
            'metric': 'mae',
            'device': 'gpu',
            'gpu_device_id': 0,
            'max_depth': 7,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1,
            'seed': 42,
        }
        
        train_dataset = lgb.Dataset(X_train, label=y_train)
        model_lgb = lgb.train(params_lgb, train_dataset, num_boost_round=300)
        
        logger.info("  ✓ LightGBM trained successfully")
        
        # Predict
        logger.info("  Making predictions...")
        predictions = model_lgb.predict(X_all)
        logger.info(f"  ✓ Predictions: {len(predictions)}")
        
    except Exception as e:
        logger.warning(f"  ✗ LightGBM failed: {e}")
        logger.info("  Trying XGBoost instead...")
        
        try:
            params_xgb = {
                'objective': 'reg:squarederror',
                'metric': 'mae',
                'tree_method': 'hist',
                'gpu_id': 0,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42,
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            model_xgb = xgb.train(params_xgb, dtrain, num_boost_round=300)
            
            logger.info("  ✓ XGBoost trained successfully")
            
            dtest = xgb.DMatrix(X_all)
            predictions = model_xgb.predict(dtest)
            logger.info(f"  ✓ Predictions: {len(predictions)}")
            
        except Exception as e2:
            logger.error(f"  ✗ Both models failed: {e2}")
            logger.info("  Using mean prediction as fallback...")
            predictions = np.full(len(X_all), y_train.mean())
    
    # ========== CREATE SUBMISSION ==========
    logger.info("\nStep 6: Creating submission CSV...")
    
    months = ['201610', '201611', '201612', '201710', '201711', '201712']
    
    submission = pd.DataFrame()
    submission['ParcelId'] = parcel_ids
    
    for month in months:
        submission[month] = predictions
    
    logger.info(f"  ✓ Submission shape: {submission.shape}")
    logger.info(f"  ✓ Columns: {submission.columns.tolist()}")
    
    # ========== SAVE SUBMISSION ==========
    logger.info("\nStep 7: Saving submission...")
    
    output_file = output_dir / 'submission.csv'
    submission.to_csv(output_file, index=False)
    
    file_size = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ Saved to: {output_file}")
    logger.info(f"  ✓ File size: {file_size:.1f} MB")
    
    # Show preview
    logger.info("\n  First 5 rows:")
    for idx, row in submission.head().iterrows():
        logger.info(f"    {row['ParcelId']}: {row['201610']:.6f}, {row['201611']:.6f}, ...")
    
    logger.info("\n" + "="*80)
    logger.info("✓ SUBMISSION READY!")
    logger.info("="*80)
    logger.info(f"\nFile: outputs/submission.csv")
    logger.info(f"Rows: {len(submission):,}")
    logger.info(f"Columns: ParcelId, 201610, 201611, 201612, 201710, 201711, 201712")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Go to: https://www.kaggle.com/c/zillow-prize-1/submissions")
    logger.info(f"2. Upload: outputs/submission.csv")
    logger.info(f"3. Click: Make Submission")
    logger.info("="*80 + "\n")
    
    return submission


if __name__ == "__main__":
    submission = generate_submission()