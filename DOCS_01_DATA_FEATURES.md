# DATA & FEATURE PROCESSING MODULES

## Table of Contents
1. [data_preparation.py](#data_preparationpy)
2. [feature_engineering.py](#feature_engineeringpy)

---

## data_preparation.py

### Module Purpose
Handles loading, cleaning, merging, and preparing data from raw CSV files into machine learning-ready format.

### Complete Code Documentation

```python
import pandas as pd
```
**Line 1**: Import pandas
- Load and manipulate DataFrames
- Read/write CSV files
- Data merging and reshaping

```python
import numpy as np
```
**Line 2**: Import numpy
- Numerical operations
- Array manipulations
- Statistical functions

```python
import logging
```
**Line 3**: Import logging
- Track data processing progress
- Log warnings and errors
- Informative console output

```python
from pathlib import Path
```
**Line 4**: Import Path
- Cross-platform file path handling
- Path operations (join, exists, mkdir)
- Better than os.path

```python
class DataPreparation:
```
**Line 7**: Define main class
- Encapsulates all data preparation logic
- Reusable across different datasets
- Methods for different processing steps

```python
    def __init__(self, data_dir='data'):
```
**Line 8**: Constructor method
- **data_dir='data'**: Default data directory path
- **self.data_dir**: Store as instance variable
- Initialize logging in constructor

```python
        self.data_dir = Path(data_dir)
```
**Line 9**: Convert string to Path object
- Enables OS-agnostic path operations
- Can use `/` operator for path joining
- Example: `self.data_dir / 'properties_2016.csv'`

```python
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
```
**Lines 10-14**: Configure logging
- **level=logging.INFO**: Show INFO and above (WARNING, ERROR, CRITICAL)
- **format**: Include timestamp, level name, message
- **self.logger**: Instance logger for this class
- All class methods use `self.logger.info()`, `self.logger.error()`, etc.

```python
    def load_properties(self, filename):
```
**Line 16**: Method to load property data
- **filename**: CSV filename (e.g., 'properties_2016.csv')
- Returns DataFrame with property features

```python
        """Load properties CSV file"""
        try:
            self.logger.info(f"Loading {filename}...")
            filepath = self.data_dir / filename
```
**Lines 17-20**: Load with error handling
- **filepath**: Full path to file
- **try block**: Catch loading errors
- Log before attempting load

```python
            properties = pd.read_csv(filepath, low_memory=False)
```
**Line 21**: Read CSV file
- **low_memory=False**: Infer dtypes for entire column (accurate but slower)
- **Returns**: DataFrame with all columns parsed correctly
- Shape: ~3M rows × 58 columns

```python
            self.logger.info(f"  ✓ Loaded {properties.shape[0]:,} rows, {properties.shape[1]} columns")
            return properties
```
**Lines 22-23**: Log success and return
- **{:,}**: Format number with comma separator
- Example: "✓ Loaded 2,985,217 rows, 58 columns"

```python
        except FileNotFoundError:
            self.logger.error(f"  ✗ File not found: {filepath}")
            return None
        except Exception as e:
            self.logger.error(f"  ✗ Error loading {filename}: {e}")
            return None
```
**Lines 24-28**: Error handling
- **FileNotFoundError**: Specific error if file doesn't exist
- **Exception**: Catch any other errors
- Return None if loading fails (calling code checks for None)

```python
    def load_training_data(self, filename):
```
**Line 30**: Method to load training labels
- **filename**: CSV filename (e.g., 'train_2016_v2.csv')
- Returns DataFrame with parcelid and logerror columns

```python
        """Load training data with target variable"""
        try:
            self.logger.info(f"Loading training data from {filename}...")
            filepath = self.data_dir / filename
            training_data = pd.read_csv(filepath)
```
**Lines 31-35**: Load training CSV
- Usually smaller files (~90K rows)
- Contains: parcelid, logerror, transactiondate

```python
            self.logger.info(f"  ✓ Loaded {training_data.shape[0]:,} training samples")
            return training_data
        except Exception as e:
            self.logger.error(f"  ✗ Error loading training data: {e}")
            return None
```
**Lines 36-39**: Log and return or error

```python
    def merge_properties_with_training(self, training_df, properties_df):
```
**Line 41**: Method to merge training labels with property features
- **training_df**: Training data with parcelid and logerror
- **properties_df**: Property features with parcelid
- Returns merged DataFrame

```python
        """Merge training data with property features"""
        self.logger.info("Merging training data with properties...")
        
        try:
            merged = training_df.merge(
                properties_df,
                on='parcelid',
                how='left'
            )
```
**Lines 42-49**: Perform SQL-like join
- **on='parcelid'**: Join on matching parcelid
- **how='left'**: Keep all rows from training_df (left table)
- Each training sample now has all property features

```python
            self.logger.info(f"  ✓ Merged shape: {merged.shape}")
            return merged
```
**Lines 50-51**: Log result

```python
        except Exception as e:
            self.logger.error(f"  ✗ Error merging data: {e}")
            return None
```
**Lines 52-54**: Error handling

```python
    def handle_missing_values(self, df, strategy='median'):
```
**Line 56**: Method to fill missing values
- **df**: DataFrame with potential NaN values
- **strategy='median'**: How to fill (median, mean, forward fill, etc.)

```python
        """Handle missing values in DataFrame"""
        self.logger.info(f"Handling missing values with {strategy}...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
```
**Lines 57-60**: Get numeric columns
- Only numeric columns can have statistical imputation
- Text/categorical columns handled differently

```python
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:  # If column has NaN
```
**Lines 61-62**: Iterate through columns with NaN
- Check if column has any NaN values
- Only process columns that need it

```python
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = df[col].mode()[0]  # Most common value
```
**Lines 63-68**: Calculate fill value
- **median()**: Middle value (robust to outliers)
- **mean()**: Average value
- **mode()[0]**: Most frequent value
- Use median by default (recommended for ML)

```python
                df[col].fillna(fill_value, inplace=True)
```
**Line 69**: Fill NaN with calculated value
- **inplace=True**: Modify DataFrame in place (don't create copy)

```python
        self.logger.info(f"  ✓ Missing values handled")
        return df
```
**Lines 70-71**: Log completion

```python
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
```
**Line 73**: Method to remove outlier rows
- **columns**: Which columns to check for outliers (default: all numeric)
- **method='iqr'**: Interquartile Range method
- **threshold=1.5**: Multiplier for IQR (1.5 = standard outlier detection)

```python
        """Remove outlier rows based on IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
```
**Lines 74-76**: Default to all numeric columns if not specified

```python
        initial_rows = len(df)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)  # 25th percentile
            Q3 = df[col].quantile(0.75)  # 75th percentile
            IQR = Q3 - Q1
```
**Lines 78-82**: Calculate IQR (Interquartile Range)
- **Q1**: 25% of values below this
- **Q3**: 75% of values below this
- **IQR**: Spread of middle 50% of data

```python
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
```
**Lines 83-84**: Define outlier bounds
- Values outside these bounds are outliers
- **threshold=1.5**: Standard statistical threshold

```python
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
```
**Line 85**: Filter to keep only non-outlier rows
- Keep rows where value is within bounds
- Remove rows with extreme values

```python
        final_rows = len(df)
        self.logger.info(f"  ✓ Removed {initial_rows - final_rows} outlier rows")
        return df
```
**Lines 86-88**: Log how many rows removed

```python
    def prepare_data(self, properties_file, training_file):
```
**Line 90**: Main orchestration method
- Combines all data preparation steps
- **properties_file**: CSV with property features
- **training_file**: CSV with training labels
- Returns X_train, X_test, y_train, y_test (or combined DataFrames)

```python
        """Complete data preparation pipeline"""
        # Load data
        properties = self.load_properties(properties_file)
        training = self.load_training_data(training_file)
```
**Lines 91-94**: Load both datasets

```python
        if properties is None or training is None:
            self.logger.error("Failed to load data")
            return None, None, None
```
**Lines 95-97**: Check if loading succeeded

```python
        # Merge
        data = self.merge_properties_with_training(training, properties)
        if data is None:
            return None, None, None
```
**Lines 99-101**: Merge training with properties

```python
        # Handle missing values
        data = self.handle_missing_values(data, strategy='median')
```
**Line 103**: Fill NaN values with median

```python
        # Remove outliers
        data = self.remove_outliers(data, method='iqr', threshold=1.5)
```
**Line 105**: Remove extreme outliers

```python
        self.logger.info(f"Data preparation complete: {data.shape}")
        return data, data, data  # X, y, combined (varies by usage)
```
**Lines 107-108**: Log final shape and return
- Returns data in different formats depending on how it's called

---

## feature_engineering.py

### Module Purpose
Create new features from raw data to improve model performance. More features = better pattern capture.

### Complete Code Documentation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import logging
```
**Lines 1-4**: Imports
- **StandardScaler**: Normalize features (mean=0, std=1)
- **PolynomialFeatures**: Generate polynomial combinations (x, x², xy, etc.)
- **logging**: Track feature engineering progress

```python
class FeatureEngineering:
    """Create features from raw data"""
```
**Lines 6-7**: Class definition

```python
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
```
**Lines 8-10**: Initialize logger

```python
    def create_interaction_features(self, df):
```
**Line 12**: Create interaction/combination features
- Multiply features together to capture relationships
- Example: price × sqft = interaction feature

```python
        """Create interaction features between key variables"""
        self.logger.info("Creating interaction features...")
        
        interactions = []
```
**Lines 13-16**: Start tracking new features

```python
        # Price × Size interaction
        if 'taxvaluedollarcnt' in df.columns and 'calculatedfinishedsquarefeet' in df.columns:
            df['price_size_interaction'] = df['taxvaluedollarcnt'] * df['calculatedfinishedsquarefeet']
            interactions.append('price_size_interaction')
```
**Lines 18-21**: Create price × size interaction
- Properties with high price AND high size = extreme value
- Captures combined effect

```python
        # Rooms × Size interaction
        if 'bedroomcnt' in df.columns and 'bathroomcnt' in df.columns:
            df['rooms_interaction'] = df['bedroomcnt'] * df['bathroomcnt']
            interactions.append('rooms_interaction')
```
**Lines 23-26**: Bedrooms × bathrooms
- More rooms + more baths = luxury property indicator

```python
        self.logger.info(f"  ✓ Created {len(interactions)} interaction features")
        return df
```
**Lines 28-29**: Log and return

```python
    def create_binned_features(self, df):
```
**Line 31**: Create categorical bins from continuous variables
- Convert continuous to discrete categories
- Example: age 0-10 years, 10-20 years, etc.

```python
        """Create binned/categorical features"""
        self.logger.info("Creating binned features...")
        
        # Age bins
        if 'yearbuilt' in df.columns:
            df['property_age'] = 2024 - df['yearbuilt']
            df['age_group'] = pd.cut(
                df['property_age'],
                bins=[0, 10, 20, 50, 100, 200],
                labels=['very_new', 'new', 'medium', 'old', 'very_old']
            )
```
**Lines 32-43**: Create property age and age groups
- **property_age**: Years since built (2024 - yearbuilt)
- **age_group**: Categorical bins (very_new, new, medium, old, very_old)
- Categorical features can be useful for tree models

```python
        # Price bins
        if 'taxvaluedollarcnt' in df.columns:
            df['price_group'] = pd.qcut(
                df['taxvaluedollarcnt'],
                q=5,
                labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                duplicates='drop'
            )
```
**Lines 45-52**: Create price categories
- **pd.qcut()**: Divide into quantiles (equal number of samples per bin)
- q=5: Create 5 equal-sized groups
- **duplicates='drop'**: Handle edge case with duplicate values

```python
        self.logger.info("  ✓ Created binned features")
        return df
```
**Lines 54-55**: Log and return

```python
    def create_aggregate_features(self, df):
```
**Line 57**: Create aggregated features by group
- Calculate statistics grouped by geographic/categorical variables
- Example: Average price by county

```python
        """Create aggregate features by county/region"""
        self.logger.info("Creating aggregate features...")
        
        # County-level statistics
        if 'fips' in df.columns and 'taxvaluedollarcnt' in df.columns:
            df['county_median_price'] = df.groupby('fips')['taxvaluedollarcnt'].transform('median')
            df['county_mean_price'] = df.groupby('fips')['taxvaluedollarcnt'].transform('mean')
            df['county_std_price'] = df.groupby('fips')['taxvaluedollarcnt'].transform('std')
```
**Lines 58-64**: County-level statistics
- **groupby('fips')**: Group by county FIPS code
- **transform()**: Return value for each original row (not aggregated)
- Each property gets county-level statistics as features

```python
            # Price relative to county
            df['price_vs_county'] = df['taxvaluedollarcnt'] / (df['county_median_price'] + 1)
```
**Line 66**: Create relative price feature
- Property price / county median price
- >1.0 = Above county average
- <1.0 = Below county average
- +1: Avoid division by zero

```python
        # Region statistics
        if 'regionidcity' in df.columns:
            df['city_count_properties'] = df.groupby('regionidcity').transform('size')
```
**Lines 69-70**: Properties per city
- How many properties in this city
- Proxy for city size/desirability

```python
        self.logger.info("  ✓ Created aggregate features")
        return df
```
**Lines 72-73**: Log and return

```python
    def scale_features(self, X_train, X_test):
```
**Line 75**: Normalize features to same scale
- **X_train**: Training features to fit scaler
- **X_test**: Test features to transform

```python
        """Standardize features using training data"""
        self.logger.info("Scaling features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
```
**Lines 76-80**: Fit and transform
- **fit_transform()**: Calculate mean/std from training, then normalize
- **transform()**: Use training stats to normalize test data (IMPORTANT!)
- Prevents test data leakage

```python
        self.logger.info("  ✓ Features scaled")
        return X_train_scaled, X_test_scaled, scaler
```
**Lines 81-82**: Return scaled data and scaler object (for future use)

```python
    def select_best_features(self, X, y, n_features=30):
```
**Line 84**: Select top N most important features
- Reduces dimensionality
- Improves training speed
- Can improve generalization

```python
        """Select top N features by correlation with target"""
        self.logger.info(f"Selecting top {n_features} features...")
        
        correlations = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                corr = X[col].corr(y)
                correlations.append((col, abs(corr)))
```
**Lines 85-91**: Calculate correlations
- For each numeric column, calculate correlation with target (y)
- **abs(corr)**: Use absolute value (both + and - correlations matter)

```python
        correlations.sort(key=lambda x: x[1], reverse=True)
        best_features = [x[0] for x in correlations[:n_features]]
```
**Lines 92-93**: Sort and select top N
- Sort by correlation descending
- Take first N features

```python
        self.logger.info(f"  ✓ Selected {len(best_features)} features")
        return best_features
```
**Lines 94-95**: Return feature names

---

End of Data & Feature Processing Module Documentation
