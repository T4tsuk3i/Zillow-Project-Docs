# CODE DOCUMENTATION

## Table of Contents

1. [generate_submission.py](#generate_submissionpy)
2. [generate_second_submission.py](#generate_second_submissionpy)
3. [validate_submission.py](#validate_submissionpy)
4. [Key Concepts Explained](#key-concepts-explained)

---

## generate_submission.py

### Full Code with Complete Line-by-Line Explanation

```python
"""
Simple Kaggle Submission Generator
Generates submission.csv from trained models
Run this AFTER main.py completes
"""
```

**Lines 1-4**: Docstring explaining the script's purpose

- When to use: After main.py has completed training
- What it generates: submission.csv file for Kaggle submission

```python
import pandas as pd
```

**Line 7**: Import pandas library

- **What**: Data manipulation library in Python
- **Why**: Need to load CSVs, create DataFrames, handle tabular data
- **What it does**: `pd` is the alias for all pandas functions
- **Examples**:
  - `pd.read_csv()` - reads CSV files
  - `pd.DataFrame()` - creates table-like data structure
  - `pd.concat()` - combines multiple DataFrames

```python
import numpy as np
```

**Line 8**: Import NumPy library

- **What**: Numerical computing library for Python
- **Why**: Need efficient array operations, statistical functions
- **What it does**: `np` is alias for numpy functions
- **Examples**:
  - `np.number` - numeric data types
  - `np.full()` - create array filled with one value
  - `np.mean()` - calculate mean

```python
import xgboost as xgb
```

**Line 9**: Import XGBoost library

- **What**: Extreme Gradient Boosting - fast tree boosting algorithm
- **Why**: One of best performing ML algorithms, has GPU support
- **What it does**: `xgb` is alias for XGBoost functions
- **Usage in this script**: `xgb.DMatrix()`, `xgb.train()`, `xgb.predict()`

```python
import lightgbm as lgb
```

**Line 10**: Import LightGBM library

- **What**: Light Gradient Boosting Machine - another boosting algorithm
- **Why**: Often faster than XGBoost, excellent performance, GPU capable
- **What it does**: `lgb` is alias for LightGBM functions
- **Usage in this script**: `lgb.Dataset()`, `lgb.train()`, `model.predict()`

```python
import logging
```

**Line 11**: Import logging module

- **What**: Python's built-in logging framework
- **Why**: Print informative messages about script progress
- **What it does**: Create loggers that print to console
- **Advantages over print()**: Structured, can log to files, can set levels (DEBUG, INFO, WARNING, ERROR)

```python
from pathlib import Path
```

**Line 12**: Import Path from pathlib

- **What**: Object-oriented way to handle file paths
- **Why**: Works across different operating systems (Windows, Mac, Linux)
- **What it does**: Create Path objects for directories/files
- **Example**: `Path('data') / 'file.csv'` automatically adds correct separator

```python
logging.basicConfig(level=logging.INFO, format='%(message)s')
```

**Line 15**: Configure logging

- **level=logging.INFO**: Only show INFO level and above (ignore DEBUG messages)
- **format='%(message)s'**: Show only the message (no timestamps, no level names)
- **Effect**: All `logger.info()` calls will print their message to console

```python
logger = logging.getLogger(__name__)
```

**Line 16**: Create logger object

- **What**: Create a logger instance for this module
- **__name__**: Name of current script (in this case, '__main__')
- **What it does**: This logger instance is used for all logging in the script
- **Example usage**: `logger.info("message")` prints the message

```python
def generate_submission():
```

**Line 19**: Define main function

- **What**: Creates a function called `generate_submission`
- **Why**: Organize code into logical blocks, reusable
- **What it does**: Everything inside this function will run when called
- **Return value**: This function returns a `submission` DataFrame at the end

```python
    logger.info("\n" + "="*80)
```

**Line 20**: Print header line with logger

- `"\n"`: New line before header
- `"="*80`: Creates string "===...===" (80 equals signs)
- **Why**: Visual separator to make output readable

```python
    logger.info("KAGGLE SUBMISSION GENERATOR")
```

**Line 21**: Print title

- Informs user what script is running

```python
    logger.info("="*80 + "\n")
```

**Line 22**: Print closing header

- Completes the visual separator

```python
    data_dir = Path('data')
```

**Line 24**: Define data directory path

- **What**: Creates Path object pointing to 'data' folder
- **Why**: Know where to find input CSV files
- **Usage**: Later use `data_dir / 'properties_2016.csv'` to build full path

```python
    output_dir = Path('outputs')
```

**Line 25**: Define output directory path

- **What**: Creates Path object pointing to 'outputs' folder
- **Why**: Know where to save submission.csv file
- **Usage**: Save results to `output_dir / 'submission.csv'`

```python
    output_dir.mkdir(exist_ok=True)
```

**Line 26**: Create output directory

- **mkdir()**: "make directory"
- **exist_ok=True**: Don't crash if directory already exists
- **Why**: Ensure output folder exists before saving files

```python
    logger.info("Step 1: Loading test properties...")
```

**Line 28**: Log current step

- Tells user what's happening next

```python
    try:
```

**Line 30**: Start error handling block

- **What**: Try to execute the following code
- **If error occurs**: Skip to `except` block instead of crashing
- **Why**: Loading files might fail (file not found, permission denied, etc.)

```python
        properties_2016 = pd.read_csv(data_dir / 'properties_2016.csv', low_memory=False)
```

**Line 31**: Load 2016 properties CSV file

- **pd.read_csv()**: Pandas function to read CSV file
- **data_dir / 'properties_2016.csv'**: Full path to file
- **low_memory=False**: Don't infer data types in chunks (slower but more accurate)
- **Result**: DataFrame with ~3M rows, 58 columns
- **Columns**: parcelid, yearbuilt, calculatedfinishedsquarefeet, taxvaluedollarcnt, etc.

```python
        logger.info(f"  ✓ 2016 properties: {properties_2016.shape}")
```

**Line 32**: Log successful load with file shape

- **f"..."**: F-string (formatted string) with variable interpolation
- **{properties_2016.shape}**: Shows (num_rows, num_columns) tuple
- **✓**: Checkmark emoji to indicate success
- **Example output**: "✓ 2016 properties: (2985217, 58)"

```python
    except Exception as e:
```

**Line 33**: Catch any error

- **Exception**: Catches any type of Python error
- **as e**: Store error details in variable `e`
- **Why**: Log error and continue execution instead of crashing

```python
        logger.error(f"  ✗ Failed to load 2016: {e}")
```

**Line 34**: Log error message

- **logger.error()**: Print error-level message
- **✗**: Cross emoji to indicate failure
- **{e}**: Print actual error message (e.g., "FileNotFoundError: properties_2016.csv not found")

```python
        properties_2016 = None
```

**Line 35**: Set to None to indicate failure

- **Why**: Later code checks if `properties_2016 is not None`
- **Effect**: Skip using 2016 data if loading failed

```python
    try:
        properties_2017 = pd.read_csv(data_dir / 'properties_2017.csv', low_memory=False)
        logger.info(f"  ✓ 2017 properties: {properties_2017.shape}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load 2017: {e}")
        properties_2017 = None
```

**Lines 37-42**: Same as 2016 but for 2017 properties

- Loads properties_2017.csv
- Sets to None if fails

```python
    # Combine
    if properties_2016 is not None and properties_2017 is not None:
        properties = pd.concat([properties_2016, properties_2017], ignore_index=True)
```

**Lines 44-46**: Combine 2016 and 2017 data

- **if...and...**: Both must be not None (both must have loaded successfully)
- **pd.concat()**: Concatenate (combine) two DataFrames
- **[list]**: Pass DataFrames in a list
- **ignore_index=True**: Reset row numbers (0 to total_rows-1)
- **Result**: Combined DataFrame with 2016 and 2017 rows

```python
    elif properties_2016 is not None:
        properties = properties_2016
```

**Lines 47-48**: If only 2016 loaded

- Use 2016 data alone

```python
    elif properties_2017 is not None:
        properties = properties_2017
```

**Lines 49-50**: If only 2017 loaded

- Use 2017 data alone

```python
    else:
        logger.error("ERROR: No properties loaded!")
        return
```

**Lines 51-53**: If neither loaded

- Log critical error
- **return**: Exit function immediately (don't proceed)
- **Why**: Can't predict without property data

```python
    # Get unique properties
    properties = properties.drop_duplicates(subset=['parcelid'], keep='first')
```

**Line 56**: Remove duplicate properties

- **drop_duplicates()**: Remove rows that are duplicates
- **subset=['parcelid']**: Consider only 'parcelid' column for duplicate detection
- **keep='first'**: If duplicate, keep first occurrence, discard rest
- **Why**: Some properties might appear in both 2016 and 2017 datasets

```python
    parcel_ids = properties['parcelid'].values
```

**Line 57**: Extract parcel IDs

- **properties['parcelid']**: Get parcelid column (Pandas Series)
- **.values**: Convert to NumPy array
- **Why**: Need array of IDs for submission CSV ParcelId column
- **Type**: NumPy array of integers, length ~3M

```python
    logger.info(f"  ✓ Total unique properties: {len(parcel_ids):,}")
```

**Line 58**: Log number of properties

- **{len(parcel_ids):,}**: Print length with comma separators
- **Example**: "✓ Total unique properties: 2,985,217"

```python
    logger.info("\nStep 2: Preparing features...")
```

**Line 60**: Log next step

```python
    # Select numeric columns only
    numeric_cols = properties.select_dtypes(include=[np.number]).columns.tolist()
```

**Line 63**: Get all numeric columns

- **select_dtypes()**: Filter columns by data type
- **include=[np.number]**: Only columns with numeric data types
- **.columns**: Get column names (Index object)
- **.tolist()**: Convert to Python list
- **Why**: Tree models work with numeric data, skip text/dates
- **Result**: List of column names like ['latitude', 'longitude', 'taxvaluedollarcnt', ...]

```python
    # Remove target-like columns
    drop_cols = {'logerror', 'parcelid', 'transactiondate'}
```

**Line 66**: Define columns to remove

- **logerror**: This is target variable (test data doesn't have it anyway)
- **parcelid**: This is ID, not feature
- **transactiondate**: Not numeric, causes errors in correlation
- **Set**: Unordered collection of unique values (like list but faster lookup)

```python
    feature_cols = [col for col in numeric_cols if col not in drop_cols]
```

**Line 67**: Create feature list

- **list comprehension**: Create list by filtering
- **for col in numeric_cols**: Iterate through each numeric column
- **if col not in drop_cols**: Keep only if NOT in drop_cols set
- **Result**: List of all numeric columns except logerror, parcelid, transactiondate
- **Example**: ['latitude', 'longitude', 'yearbuilt', 'bedroomcnt', ...]

```python
    logger.info(f"  ✓ Features available: {len(feature_cols)}")
```

**Line 69**: Log feature count

- **Example output**: "✓ Features available: 54"

```python
    # Fill missing values with median
    X_all = properties[feature_cols].copy()
```

**Line 72**: Create feature matrix

- **properties[feature_cols]**: Select only feature columns
- **.copy()**: Make independent copy (don't modify original)
- **Result**: DataFrame with ~3M rows, 54 columns
- **Why copy**: Avoid "SettingWithCopyWarning" when modifying

```python
    for col in feature_cols:
        if X_all[col].isnull().sum() > 0:
            X_all[col] = X_all[col].fillna(X_all[col].median())
```

**Lines 73-75**: Fill missing values

- **for col in feature_cols**: Iterate through each feature
- **X_all[col].isnull()**: Boolean array (True where NaN, False elsewhere)
- **.sum()**: Count number of True values (number of NaNs)
- **> 0**: If there are any NaNs
- **X_all[col].fillna()**: Replace NaNs with value
- **X_all[col].median()**: Median of non-NaN values
- **Why median**: Robust to outliers (better than mean)

```python
    logger.info(f"  ✓ Features shape: {X_all.shape}")
    logger.info(f"  ✓ Missing values: {X_all.isnull().sum().sum()}")
```

**Lines 77-78**: Log data shape and missing value count

- **{X_all.shape}**: Print (rows, columns) - e.g., "(2985217, 54)"
- **{X_all.isnull().sum().sum()}**: Count total NaNs remaining
- **Should be 0**: All NaNs should be filled

```python
    logger.info("\nStep 3: Loading training data for model training...")
```

**Line 80**: Log next step

```python
    try:
        train_2016 = pd.read_csv(data_dir / 'train_2016_v2.csv')
        logger.info(f"  ✓ Train 2016: {train_2016.shape}")
    except:
        train_2016 = None
```

**Lines 82-86**: Load 2016 training labels

- **Columns**: parcelid, logerror, transactiondate
- **Shape**: (90,275, 3) - 90K samples with logerror values
- **Purpose**: These are the target values we trained models on
- **Silent except**: If fails, just set to None (no logging)

```python
    try:
        train_2017 = pd.read_csv(data_dir / 'train_2017.csv')
        logger.info(f"  ✓ Train 2017: {train_2017.shape}")
    except:
        train_2017 = None
```

**Lines 88-92**: Load 2017 training labels

- Same as 2016 but for 2017 data
- **Shape**: (77,613, 3) - 77K samples

```python
    # Merge with properties
    if train_2016 is not None:
        train_2016 = train_2016.merge(properties_2016, on='parcelid', how='left')
```

**Lines 95-97**: Merge training labels with properties

- **merge()**: Join two DataFrames like SQL JOIN
- **on='parcelid'**: Join on matching parcelid values
- **how='left'**: Keep all rows from left (train_2016)
- **Result**: Each training sample now has all its features from properties
- **Shape**: (90,275, 3+58) = (90,275, 61) - logerror + all property features

```python
    if train_2017 is not None:
        train_2017 = train_2017.merge(properties_2017, on='parcelid', how='left')
```

**Lines 99-100**: Same for 2017 data

```python
    # Combine training data
    if train_2016 is not None and train_2017 is not None:
        train_data = pd.concat([train_2016, train_2017], ignore_index=True)
    elif train_2016 is not None:
        train_data = train_2016
    else:
        train_data = train_2017
```

**Lines 102-107**: Combine 2016 and 2017 training data

- Same logic as properties combining
- **Result**: DataFrame with ~168K samples (90K + 77K)

```python
    logger.info(f"  ✓ Combined training data: {train_data.shape}")
```

**Line 109**: Log combined training data shape

```python
    logger.info("\nStep 4: Preparing training data...")
```

**Line 111**: Log next step

```python
    X_train = train_data[feature_cols].copy()
```

**Line 113**: Extract training features

- Select same features as test set
- **X_train**: Features for training, shape (168K, 54)

```python
    y_train = train_data['logerror'].copy()
```

**Line 114**: Extract training targets

- **y_train**: Target values (logerror), shape (168K,)

```python
    # Fill missing values
    for col in feature_cols:
        if X_train[col].isnull().sum() > 0:
            X_train[col] = X_train[col].fillna(X_train[col].median())
```

**Lines 116-118**: Fill missing values in training set

- Same strategy as test set

```python
    # Remove any NaN rows
    valid_idx = ~(X_train.isnull().any(axis=1) | y_train.isnull())
```

**Line 121**: Create boolean array of valid rows

- **X_train.isnull()**: Boolean DataFrame (True where NaN)
- **.any(axis=1)**: True if ANY column in row is NaN
- **y_train.isnull()**: Boolean array (True where target is NaN)
- **|**: OR operator (True if either is True)
- **~**: NOT operator (flip True/False)
- **valid_idx**: Boolean array (True where row is completely valid)

```python
    X_train = X_train[valid_idx]
    y_train = y_train[valid_idx]
```

**Lines 122-123**: Keep only valid rows

- Filter DataFrames to remove rows with ANY NaN values
- **Result**: Smaller training set, completely clean

```python
    logger.info(f"  ✓ Training data: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  ✓ Missing in training: {X_train.isnull().sum().sum()}")
```

**Lines 125-126**: Log training data info

- **Should show**: X=(167888, 54), y=(167888,)
- **Should show**: Missing in training: 0

```python
    logger.info("\nStep 5: Training LightGBM on full training data...")
```

**Line 128**: Log next step

```python
    try:
```

**Line 130**: Start error handling for LightGBM training

```python
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
```

**Lines 131-143**: Define LightGBM hyperparameters

- **'objective': 'regression'**: Predict continuous values (not classification)
- **'metric': 'mae'**: Optimize for Mean Absolute Error
- **'device': 'gpu'**: Use GPU acceleration
- **'gpu_device_id': 0**: Use first GPU (RTX 3050)
- **'max_depth': 7**: Maximum tree depth (7 layers)
- **'num_leaves': 31**: Maximum leaves per tree
- **'learning_rate': 0.05**: Step size for boosting (smaller = slower learning but more accurate)
- **'subsample': 0.8**: Use 80% of rows when building each tree
- **'colsample_bytree': 0.8**: Use 80% of columns for each tree
- **'verbose': -1**: No progress output (silent)
- **'seed': 42**: Random seed for reproducibility

```python
        train_dataset = lgb.Dataset(X_train, label=y_train)
```

**Line 145**: Create LightGBM Dataset

- **lgb.Dataset()**: Special data structure optimized for training
- **X_train**: Features
- **label=y_train**: Target values
- **Why**: LightGBM works faster with its own data format

```python
        model_lgb = lgb.train(params_lgb, train_dataset, num_boost_round=300)
```

**Line 146**: Train LightGBM model

- **lgb.train()**: Main training function
- **params_lgb**: Hyperparameters dict
- **train_dataset**: Training data
- **num_boost_round=300**: Number of boosting iterations (trees to build)
- **Result**: Trained model object
- **Time**: ~5-10 minutes on GPU

```python
        logger.info("  ✓ LightGBM trained successfully")
```

**Line 148**: Log success

```python
        # Predict
        logger.info("  Making predictions...")
        predictions = model_lgb.predict(X_all)
```

**Lines 150-151**: Make predictions on all test properties

- **model_lgb.predict()**: Use trained model to predict
- **X_all**: Test features (3M properties, 54 features)
- **predictions**: Array of shape (3M,) with predicted logerror values
- **Example values**: 0.0345, -0.0123, 0.0567, ...
- **Time**: ~30 seconds to predict 3M properties

```python
        logger.info(f"  ✓ Predictions: {len(predictions)}")
```

**Line 152**: Log prediction count

```python
    except Exception as e:
        logger.warning(f"  ✗ LightGBM failed: {e}")
        logger.info("  Trying XGBoost instead...")
```

**Lines 154-156**: Catch LightGBM errors and try fallback

- If LightGBM fails (GPU not available, out of memory, etc.)
- Try XGBoost as backup

```python
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
```

**Lines 158-167**: Define XGBoost hyperparameters

- **'objective': 'reg:squarederror'**: Mean squared error objective
- **'metric': 'mae'**: But evaluate with MAE
- **'tree_method': 'hist'**: Histogram-based tree building (GPU-compatible)
- **'gpu_id': 0**: Use GPU 0
- **'max_depth': 6**: Maximum depth (shallower than LightGBM)
- **'learning_rate': 0.05**: Same as LightGBM
- **'subsample': 0.8**: 80% row sampling
- **'colsample_bytree': 0.8**: 80% column sampling
- **'seed': 42**: Reproducibility seed

```python
            dtrain = xgb.DMatrix(X_train, label=y_train)
```

**Line 169**: Create XGBoost DMatrix

- **xgb.DMatrix()**: XGBoost's data format
- Optimized for XGBoost training

```python
            model_xgb = xgb.train(params_xgb, dtrain, num_boost_round=300)
```

**Line 170**: Train XGBoost model

- **xgb.train()**: Main training function
- Same parameters as LightGBM version

```python
            logger.info("  ✓ XGBoost trained successfully")
```

**Line 172**: Log success

```python
            dtest = xgb.DMatrix(X_all)
            predictions = model_xgb.predict(dtest)
```

**Lines 174-175**: Make predictions with XGBoost

- **xgb.DMatrix(X_all)**: Convert test features to DMatrix
- **model_xgb.predict()**: Get predictions

```python
            logger.info(f"  ✓ Predictions: {len(predictions)}")
```

**Line 176**: Log prediction count

```python
        except Exception as e2:
            logger.error(f"  ✗ Both models failed: {e2}")
            logger.info("  Using mean prediction as fallback...")
            predictions = np.full(len(X_all), y_train.mean())
```

**Lines 178-181**: Last resort fallback

- If both LightGBM and XGBoost fail
- Use training mean as prediction for all properties
- **np.full()**: Create array filled with one value
- **len(X_all)**: 3M (length of predictions needed)
- **y_train.mean()**: Average of all training targets

```python
    logger.info("\nStep 6: Creating submission CSV...")
```

**Line 183**: Log next step

```python
    months = ['201610', '201611', '201612', '201710', '201711', '201712']
```

**Line 185**: Define month columns

- **201610**: October 2016
- **201611**: November 2016
- **201612**: December 2016
- **201710**: October 2017
- **201711**: November 2017
- **201712**: December 2017

```python
    submission = pd.DataFrame()
    submission['ParcelId'] = parcel_ids
```

**Lines 187-188**: Create submission DataFrame and add parcel IDs

- **pd.DataFrame()**: Create empty DataFrame
- **submission['ParcelId']**: Add column with parcel IDs

```python
    for month in months:
        submission[month] = predictions
```

**Lines 190-191**: Add prediction columns for each month

- **for month in months**: Iterate through each month string
- **submission[month]**: Add new column
- **= predictions**: Fill with same predictions for all months
- **Why same for all months**: logerror doesn't depend on month (it's a property characteristic, not temporal)

```python
    logger.info(f"  ✓ Submission shape: {submission.shape}")
    logger.info(f"  ✓ Columns: {submission.columns.tolist()}")
```

**Lines 193-194**: Log submission DataFrame info

- **{submission.shape}**: Should be (2985217, 7) - 3M rows, 7 columns
- **{submission.columns.tolist()}**: Show column names

```python
    logger.info("\nStep 7: Saving submission...")
```

**Line 196**: Log final step

```python
    output_file = output_dir / 'submission.csv'
```

**Line 198**: Create output file path

- **output_dir / 'submission.csv'**: Combine paths (outputs/submission.csv)

```python
    submission.to_csv(output_file, index=False)
```

**Line 199**: Save DataFrame to CSV

- **to_csv()**: Pandas function to write CSV
- **output_file**: Path to save
- **index=False**: Don't write row numbers

```python
    file_size = output_file.stat().st_size / (1024 * 1024)
```

**Line 201**: Calculate file size in MB

- **output_file.stat()**: Get file statistics
- **.st_size**: File size in bytes
- **/ (1024 * 1024)**: Convert bytes to megabytes

```python
    logger.info(f"  ✓ Saved to: {output_file}")
    logger.info(f"  ✓ File size: {file_size:.1f} MB")
```

**Lines 202-203**: Log file info

- **{output_file}**: Full path
- **{file_size:.1f}**: File size rounded to 1 decimal place

```python
    # Show preview
    logger.info("\n  First 5 rows:")
    for idx, row in submission.head().iterrows():
        logger.info(f"    {row['ParcelId']}: {row['201610']:.6f}, {row['201611']:.6f}, ...")
```

**Lines 205-207**: Show preview of first 5 rows

- **.head()**: Get first 5 rows
- **.iterrows()**: Iterate through rows (idx=row number, row=data)
- **{row['ParcelId']}**: Parcel ID
- **{row['201610']:.6f}**: Prediction rounded to 6 decimals
- Example: "10754147: 0.034562, 0.034562, ..."

```python
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
```

**Lines 209-219**: Print final summary

- Visual separator with =====
- File path
- Number of rows with comma separator
- Column names
- Instructions for Kaggle upload

```python
    return submission
```

**Line 221**: Return submission DataFrame

- Allows calling code to use the submission if needed

```python
if __name__ == "__main__":
    submission = generate_submission()
```

**Lines 223-224**: Entry point

- **if __name__ == "__main__"**: Only runs if script executed directly (not imported)
- Calls the generate_submission() function
- Captures returned DataFrame in `submission` variable (though not used after)

---

## generate_second_submission.py

### Key Differences from generate_submission.py

This script is similar to `generate_submission.py` but with important optimizations:

```python
def generate_second_submission():
```

Different function name (not called `generate_submission`)

```python
    params_lgb = {
        'objective': 'regression',
        'metric': 'mae',
        'device': 'gpu',
        'gpu_device_id': 0,
        'max_depth': 12,              # ↑ INCREASED from 7 (deeper trees)
        'num_leaves': 511,            # ↑ INCREASED from 31 (more leaves)
        'learning_rate': 0.008,       # ↓ DECREASED from 0.05 (slower = better)
        'subsample': 0.7,             # ↓ DECREASED from 0.8
        'colsample_bytree': 0.7,      # ↓ DECREASED from 0.8
        'min_child_samples': 3,       # ↓ NEW: REDUCED for aggressive splitting
        'reg_alpha': 0,               # ↑ NEW: NO L1 regularization
        'reg_lambda': 0.0005,         # ↓ CHANGED: Very light L2 regularization
        'min_split_gain': 0,          # ↑ NEW: Aggressive tree splitting
        'verbose': -1,
        'seed': 43,                   # Different seed
    }
```

**HYPERPARAMETER IMPROVEMENTS**:

- **max_depth: 7→12**: Deeper trees capture more patterns
- **num_leaves: 31→511**: More leaf nodes = more flexibility
- **learning_rate: 0.05→0.008**: Slower learning = captures more nuance
- **reg_alpha: 0**: Removes L1 regularization (damping)
- **reg_lambda: 0.0005**: Very light L2 (allows variance capture)
- **min_child_samples: 3**: Allows aggressive splits

```python
    model_lgb = lgb.train(params_lgb, train_dataset, num_boost_round=600)
```

**600 boosting rounds** (instead of 300)

- More iterations = more refinement

```python
    params_xgb = {
        'objective': 'reg:squarederror',
        'metric': 'mae',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'max_depth': 12,              # ↑ INCREASED from 6
        'learning_rate': 0.008,       # ↓ DECREASED from 0.05
        'subsample': 0.7,             # ↓ DECREASED from 0.8
        'colsample_bytree': 0.7,      # ↓ DECREASED from 0.8
        'min_child_weight': 1,        # Allow small splits
        'gamma': 0,                   # No split penalty
        'reg_alpha': 0,               # NO L1
        'reg_lambda': 0.0005,         # Very light L2
        'seed': 43,
    }
```

**Similar optimizations for XGBoost**

```python
    # ===== ENSEMBLE BLENDING (MAE/RMSE OPTIMIZED) =====
    # Weight LightGBM higher because it typically has slightly better MAE
    final_pred = 0.55 * lgb_pred + 0.45 * xgb_pred
```

**Ensemble weighting**:

- **0.55 * LightGBM**: 55% weight (better MAE typically)
- **0.45 * XGBoost**: 45% weight
- **Why**: Different algorithms capture different patterns

```python
    output_file = output_dir / 'submission2.csv'  # ← NOTE: submission2.csv
```

**Different output filename**: `submission2.csv` instead of `submission.csv`

---

## validate_submission.py

### Complete Line-by-Line Explanation

```python
"""
Submission CSV Validator
Tests submission.csv predictions against actual logerror values
Calculates MAE, RMSE, and R² metrics
"""
```

Purpose: Evaluate quality of submission against training labels

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from pathlib import Path
```

**Imports**:

- **pandas**: Data manipulation
- **numpy**: Numerical arrays
- **sklearn.metrics**: Evaluation metrics
  - **mean_absolute_error**: MAE = average |prediction - actual|
  - **mean_squared_error**: MSE = average (prediction - actual)²
  - **r2_score**: R² = fraction of variance explained
- **logging**: Progress messages
- **pathlib**: File paths

```python
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
```

Configure logging (same as before)

```python
def validate_submission():
    """Validate submission CSV and calculate metrics"""
  
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION CSV VALIDATOR")
    logger.info("="*80 + "\n")
```

Print header

```python
    data_dir = Path('data')
    output_dir = Path('outputs')
```

Define data directories

```python
    # ========== LOAD SUBMISSION ==========
    logger.info("Step 1: Loading submission.csv...")
  
    submission_file = output_dir / 'submission.csv'
    if not submission_file.exists():
        logger.error(f"  ✗ File not found: {submission_file}")
        logger.error("  Please run: python generate_submission_simple.py")
        return
```

Check if submission file exists

- **if not submission_file.exists()**: File doesn't exist
- Exit if file not found

```python
    try:
        submission = pd.read_csv(submission_file)
        logger.info(f"  ✓ Submission loaded: {submission.shape}")
        logger.info(f"  ✓ Columns: {submission.columns.tolist()}")
    except Exception as e:
        logger.error(f"  ✗ Failed to load submission: {e}")
        return
```

Load submission CSV

- Print shape (should be 2985217, 7)
- Print columns

```python
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
```

Load training labels (logerror + parcelid)

```python
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
```

Combine 2016 and 2017 training data

```python
    # ========== MERGE AND EXTRACT ==========
    logger.info("\nStep 3: Merging submission with actual values...")
  
    # Rename ParcelId to parcelid for merge (if needed)
    submission_merged = submission.copy()
    if 'ParcelId' in submission_merged.columns:
        submission_merged['parcelid'] = submission_merged['ParcelId']
```

Copy submission and standardize column name to 'parcelid'

```python
    train_data['parcelid'] = train_data['parcelid'].astype(int)
    submission_merged['parcelid'] = submission_merged['parcelid'].astype(int)
```

Convert parcelid to integers for merging

```python
    # Get predictions (use first month as representative)
    if '201610' in submission_merged.columns:
        submission_merged['prediction'] = submission_merged['201610']
    else:
        logger.error("  ✗ No month columns found in submission!")
        return
```

Extract predictions from first month (201610)

- All months have same predictions, so just use first

```python
    # Merge
    merged = train_data.merge(submission_merged[['parcelid', 'prediction']], on='parcelid', how='inner')
  
    logger.info(f"  ✓ Merged data: {merged.shape}")
    logger.info(f"  ✓ Actual logerror samples: {len(merged)}")
  
    if len(merged) == 0:
        logger.error("  ✗ No matching ParcelIds between submission and training data!")
        return
```

Merge training labels with predictions

- **how='inner'**: Keep only matching parcelids
- **Result**: DataFrame with actual logerror + predicted values
- Check if any matches found

```python
    # ========== EXTRACT PREDICTIONS AND ACTUAL =====
    logger.info("\nStep 4: Extracting predictions and actual values...")
  
    y_actual = merged['logerror'].values
    y_pred = merged['prediction'].values
  
    logger.info(f"  ✓ Actual values: {len(y_actual)}")
    logger.info(f"  ✓ Predictions: {len(y_pred)}")
    logger.info(f"  ✓ Actual logerror range: [{y_actual.min():.6f}, {y_actual.max():.6f}]")
    logger.info(f"  ✓ Predicted logerror range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
```

Extract actual and predicted values as NumPy arrays

- Show ranges: actual (-4.66 to 5.26), predicted (-0.53 to 1.10)

```python
    # ========== CALCULATE METRICS =====
    logger.info("\nStep 5: Calculating metrics...")
  
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
```

Calculate three key metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: R-squared (variance explained)

```python
    # Additional metrics
    mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-10))) * 100
    median_ae = np.median(np.abs(y_actual - y_pred))
```

Calculate extra metrics

- **MAPE**: Mean Absolute Percentage Error
- **median_ae**: Median Absolute Error

```python
    logger.info(f"\n  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.6f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  Median AE: {median_ae:.6f}")
```

Print metrics

```python
    # ========== COMPARISON WITH BASELINE =====
    logger.info("\nStep 6: Baseline comparison...")
  
    # Baseline: predict mean
    y_mean = np.full_like(y_actual, y_actual.mean())
    mae_baseline = mean_absolute_error(y_actual, y_mean)
    rmse_baseline = np.sqrt(mean_squared_error(y_actual, y_mean))
    r2_baseline = r2_score(y_actual, y_mean)
```

Calculate baseline metrics

- **Baseline**: Predict same value for all (training mean)
- **y_mean**: Array filled with mean value
- **Purpose**: Compare model to stupid baseline

```python
    logger.info(f"\n  Baseline (predicting mean):")
    logger.info(f"    MAE:  {mae_baseline:.6f}")
    logger.info(f"    RMSE: {rmse_baseline:.6f}")
    logger.info(f"    R²:   {r2_baseline:.6f}")
  
    logger.info(f"\n  Your Model:")
    logger.info(f"    MAE:  {mae:.6f} (baseline: {mae_baseline:.6f}, better by {(1 - mae/mae_baseline)*100:.1f}%)")
    logger.info(f"    RMSE: {rmse:.6f} (baseline: {rmse_baseline:.6f}, better by {(1 - rmse/rmse_baseline)*100:.1f}%)")
    logger.info(f"    R²:   {r2:.6f} (baseline: {r2_baseline:.6f}, improvement: {r2 - r2_baseline:.6f})")
```

Compare your model to baseline

- Show percentage improvements

```python
    # ========== ERROR DISTRIBUTION =====
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
```

Analyze error distribution

- Show quartiles and percentiles

```python
    # ========== PERFORMANCE BY MAGNITUDE =====
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
```

Show performance by property value range

- Segment properties by absolute logerror magnitude
- Show separate metrics for each segment

```python
    # ========== SAVE RESULTS =====
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
```

Save validation metrics to CSV

- Columns: Metric name, Your value, Baseline value

```python
    # Save detailed predictions
    detailed_df = merged[['parcelid', 'logerror', 'prediction']].copy()
    detailed_df['error'] = np.abs(detailed_df['logerror'] - detailed_df['prediction'])
    detailed_df.to_csv(output_dir / 'submission_detailed_predictions.csv', index=False)
    logger.info(f"  ✓ Detailed predictions saved to: submission_detailed_predictions.csv")
```

Save detailed predictions CSV

- Columns: parcelid, actual logerror, predicted, absolute error

```python
    # ========== SUMMARY =====
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
  
    logger.info(f"\nSubmission File: {submission_file}")
    logger.info(f"Training Samples: {len(merged)}")
    logger.info(f"\nMetrics:")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²:   {r2:.6f}")
```

Print summary section

```python
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
```

Evaluate performance and show quality assessment

```python
    return results_df, detailed_df
```

Return DataFrames for further analysis if needed

```python
if __name__ == "__main__":
    validate_submission()
```

Entry point - run when script executed directly

---

## Key Concepts Explained

### What is MAE?

**Mean Absolute Error** = Average of absolute errors

```
Formula: MAE = (1/n) * Σ|y_actual - y_pred|

Example:
Predictions: [0.034, 0.032, 0.035]
Actuals:     [0.040, 0.030, 0.033]
Errors:      [0.006, 0.002, 0.002]
MAE = (0.006 + 0.002 + 0.002) / 3 = 0.003
```

**Why MAE?**

- Easy to interpret (same units as target)
- Robust to outliers
- Commonly used for regression

---

### What is RMSE?

**Root Mean Squared Error** = Square root of average squared errors

```
Formula: RMSE = sqrt((1/n) * Σ(y_actual - y_pred)²)

Example:
Predictions: [0.034, 0.032, 0.035]
Actuals:     [0.040, 0.030, 0.033]
Squared Errors: [0.000036, 0.000004, 0.000004]
MSE = (0.000036 + 0.000004 + 0.000004) / 3 = 0.000015
RMSE = sqrt(0.000015) = 0.00387
```

**Why RMSE?**

- Penalizes large errors more than MAE
- Useful when outliers are important

---

### What is R²?

**R-squared** = Fraction of variance explained by model

```
Formula: R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(y_actual - y_pred)² (residual sum of squares)
- SS_tot = Σ(y_actual - mean(y))² (total sum of squares)

Example:
If SS_res = 100 and SS_tot = 1000
R² = 1 - (100/1000) = 0.9 (explains 90% of variance)

If SS_res = 1000 and SS_tot = 1000
R² = 1 - (1000/1000) = 0 (no variance explained)
```

**Why R²?**

- Measures how well model captures patterns
- Range: -∞ to 1.0
- 1.0 = perfect predictions
- 0.0 = predicting mean (no skill)
- Negative = worse than mean

---

### GPU vs CPU Training

**CPU (Central Processing Unit)**:

- General purpose processor
- Good for: Sequential logic, small operations
- Speed: Slow for ML (maybe 5-10 million operations/sec)
- Training time: 60-75 minutes

**GPU (Graphics Processing Unit)**:

- Specialized for parallel computations
- Good for: Matrix operations, 1000s of operations in parallel
- Speed: Fast for ML (maybe 1 trillion operations/sec)
- Training time: 20-30 minutes
- RTX 3050: 2560 cores, 4GB memory

**Why GPU?**

- Tree boosting (XGBoost, LightGBM) involves matrix operations
- GPUs can do 100+ matrix rows in parallel
- 2-5x speedup on large datasets

---

### What is Gradient Boosting?

**Concept**: Build many weak models (trees) sequentially

1. Train first tree on all data
2. Calculate errors
3. Train second tree to predict errors of first
4. Combine all trees: Final prediction = sum of all tree predictions

**Example**:

```
Property 1 actual value: 100
Tree 1 prediction: 85 (error = 15)
Tree 2 fixes: predicts 10
Tree 3 fixes: predicts 5
...
Final = 85 + 10 + 5 + ... = 100 (close to actual!)
```

**Why XGBoost/LightGBM?**

- Very accurate (wins many Kaggle competitions)
- Fast training
- GPU support
- Handles categorical features
- Built-in regularization

---

### Data Flow in Submission Generation

```
1. Load Properties (3M rows, 58 columns)
   ├─ 2016 properties (2.9M)
   └─ 2017 properties (2.9M)
   
2. Load Training Data (168K samples with labels)
   ├─ 2016 training (90K)
   └─ 2017 training (77K)
   
3. Prepare Features
   ├─ Select numeric columns (54 features)
   ├─ Remove missing values
   └─ Create feature matrices (X_train, X_all)
   
4. Train Models on GPU
   ├─ LightGBM (300-600 rounds)
   └─ XGBoost (300-600 rounds)
   
5. Generate Predictions
   ├─ Predict for 3M test properties
   └─ Get 3M prediction values
   
6. Create Submission CSV
   ├─ ParcelId column (3M)
   ├─ 201610 column (3M predictions)
   ├─ 201611 column (same 3M)
   └─ ... 6 months total
   
7. Save to submission.csv (95MB)
```

---

**End of Code Documentation**

Generated: October 31, 2025
