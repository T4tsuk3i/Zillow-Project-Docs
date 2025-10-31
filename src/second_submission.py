"""
Second Kaggle Submission Generator
Creates submission2.csv with a variant LightGBM + XGBoost ensemble
Uses GPU acceleration
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_second_submission():
    """Generate Kaggle submission CSV with a different ensemble"""

    logger.info("\n" + "="*80)
    logger.info("SECOND KAGGLE SUBMISSION GENERATOR")
    logger.info("="*80 + "\n")

    data_dir = Path('data')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Load property data
    logger.info("Loading test properties...")
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

    # Deduplicate parcelid
    properties = properties.drop_duplicates(subset=['parcelid'], keep='first')
    parcel_ids = properties['parcelid'].values
    logger.info(f"  ✓ Total unique properties: {len(parcel_ids):,}")

    # Prepare features
    logger.info("\nPreparing features...")
    numeric_cols = properties.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {'logerror', 'parcelid', 'transactiondate'}
    feature_cols = [col for col in numeric_cols if col not in drop_cols]

    logger.info(f"  ✓ Features available: {len(feature_cols)}")
    X_all = properties[feature_cols].copy()
    for col in feature_cols:
        if X_all[col].isnull().sum() > 0:
            X_all[col] = X_all[col].fillna(X_all[col].median())

    logger.info(f"  ✓ Features shape: {X_all.shape}")
    logger.info(f"  ✓ Missing values: {X_all.isnull().sum().sum()}")

    # Load training data
    logger.info("\nLoading training data for model training...")
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

    if train_2016 is not None:
        train_2016 = train_2016.merge(properties_2016, on='parcelid', how='left')
    if train_2017 is not None:
        train_2017 = train_2017.merge(properties_2017, on='parcelid', how='left')

    if train_2016 is not None and train_2017 is not None:
        train_data = pd.concat([train_2016, train_2017], ignore_index=True)
    elif train_2016 is not None:
        train_data = train_2016
    else:
        train_data = train_2017

    # Prepare training data
    logger.info("\nPreparing training data...")
    X_train = train_data[feature_cols].copy()
    y_train = train_data['logerror'].copy()

    for col in feature_cols:
        if X_train[col].isnull().sum() > 0:
            X_train[col] = X_train[col].fillna(X_train[col].median())

    valid_idx = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]

    logger.info(f"  ✓ Training data: X={X_train.shape}, y={y_train.shape}")

    # Train LightGBM with different hyperparameters
    logger.info("\nTraining LightGBM (variant)...")
    try:
        params_lgb = {
            'objective': 'regression',
            'metric': 'mae',
            'device': 'gpu',
            'gpu_device_id': 0,
            'max_depth': 12,
            'num_leaves': 511,
            'learning_rate': 0.008,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_samples': 3,
            'reg_alpha': 0,
            'reg_lambda': 0.0005,
            'min_split_gain': 0,
            'verbose': -1,
            'seed': 43,
        }

        train_dataset = lgb.Dataset(X_train, label=y_train)
        model_lgb = lgb.train(params_lgb, train_dataset, num_boost_round=600)

        lgb_pred = model_lgb.predict(X_all)
        logger.info("  ✓ LightGBM (variant) trained successfully")
    except Exception as e:
        logger.error(f"  ✗ LightGBM variant training failed: {e}")
        lgb_pred = np.zeros(len(X_all))

    # Train XGBoost with different hyperparameters
    logger.info("\nTraining XGBoost (variant)...")
    try:
        params_xgb = {
            'objective': 'reg:squarederror',
            'metric': 'mae',
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'max_depth': 12,
            'learning_rate': 0.008,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 0.0005,
            'seed': 43,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        model_xgb = xgb.train(params_xgb, dtrain, num_boost_round=600)

        dtest = xgb.DMatrix(X_all)
        xgb_pred = model_xgb.predict(dtest)
        logger.info("  ✓ XGBoost (variant) trained successfully")
    except Exception as e:
        logger.error(f"  ✗ XGBoost variant training failed: {e}")
        xgb_pred = np.zeros(len(X_all))

    # Blend predictions (MAE and RMSE optimized)
    final_pred = 0.55 * lgb_pred + 0.45 * xgb_pred
    logger.info("Blended lightgbm (55%) and xgboost (45%) predictions")

    # Create submission DataFrame
    months = ['201610', '201611', '201612', '201710', '201711', '201712']
    submission = pd.DataFrame()
    submission['ParcelId'] = parcel_ids
    for month in months:
        submission[month] = final_pred

    # Save submission
    output_file = output_dir / 'submission2.csv'
    submission.to_csv(output_file, index=False)
    logger.info(f"\n✓ Submission2 csv saved: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    return submission


if __name__ == "__main__":
    generate_second_submission()
