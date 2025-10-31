"""
Data Preparation Module for Zillow Prize
Handles data loading, cleaning, and preprocessing
GPU-Compatible (PyTorch)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.missing_value_stats = {}
        
    def load_data(self, properties_file, train_file):
        """Load and merge properties and training data"""
        logger.info(f"Loading properties from {properties_file}")
        properties = pd.read_csv(self.data_dir / properties_file)
        
        logger.info(f"Loading training data from {train_file}")
        train = pd.read_csv(self.data_dir / train_file)
        
        # Merge properties with training data on parcelid
        data = train.merge(properties, on='parcelid', how='left')
        logger.info(f"Merged data shape: {data.shape}")
        
        return data, properties, train
    
    def analyze_missing_values(self, df):
        """Analyze and log missing value patterns"""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_stats = pd.DataFrame({
            'column': missing_pct.index,
            'missing_pct': missing_pct.values
        }).sort_values('missing_pct', ascending=False)
        
        self.missing_value_stats = missing_stats
        logger.info("\nTop 20 Features with Missing Values:")
        logger.info(missing_stats.head(20).to_string())
        
        return missing_stats
    
    def drop_high_missing_features(self, df, threshold=0.50):
        """Drop features with missing percentage above threshold"""
        missing_pct = (df.isnull().sum() / len(df))
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        logger.info(f"Dropping {len(cols_to_drop)} features with >{threshold*100}% missing")
        df = df.drop(columns=cols_to_drop)
        
        return df
    
    def handle_missing_values(self, df, method='median'):
        """Handle missing values with appropriate strategy"""
        logger.info(f"Handling missing values using {method} strategy")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and id columns from imputation
        numerical_cols = [col for col in numerical_cols 
                         if col not in ['logerror', 'parcelid', 'transactiondate']]
        
        if method == 'median':
            imputer = SimpleImputer(strategy='median')
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        # Handle categorical missing with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info(f"Remaining missing values: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['transactiondate']]
        
        logger.info(f"Encoding {len(categorical_cols)} categorical features")
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def feature_engineering(self, df):
        """Create engineered features from existing ones"""
        logger.info("Creating engineered features")
        
        # Age of property
        if 'yearbuilt' in df.columns:
            df['property_age'] = 2017 - df['yearbuilt']
            df.loc[df['property_age'] < 0, 'property_age'] = np.nan
        
        # Price per sqft ratio
        if 'calculatedfinishedsquarefeet' in df.columns and 'taxvaluedollarcnt' in df.columns:
            df['price_per_sqft'] = df['taxvaluedollarcnt'] / (df['calculatedfinishedsquarefeet'] + 1)
            df['price_per_sqft'] = df['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
        
        # Bedroom to bathroom ratio
        if 'bedroomcnt' in df.columns and 'bathroomcnt' in df.columns:
            df['bed_bath_ratio'] = df['bedroomcnt'] / (df['bathroomcnt'] + 1)
        
        # Total rooms
        if 'bedroomcnt' in df.columns and 'bathroomcnt' in df.columns:
            df['total_rooms'] = df['bedroomcnt'] + df['bathroomcnt']
        
        logger.info(f"Created engineered features. New shape: {df.shape}")
        return df
    
    def remove_outliers(self, df, target='logerror', iqr_multiplier=1.5):
        """Remove outliers using IQR method"""
        if target not in df.columns:
            return df
        
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        before = len(df)
        df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]
        after = len(df)
        
        logger.info(f"Removed {before - after} outliers ({(before-after)/before*100:.2f}%)")
        return df
    
    def scale_features(self, df, target='logerror', fit=True):
        """Scale numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in [target, 'parcelid']]
        
        logger.info(f"Scaling {len(numerical_cols)} numerical features")
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_data(self, properties_file, train_file, target='logerror'):
        """Complete data preparation pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING DATA PREPARATION PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        data, properties, train = self.load_data(properties_file, train_file)
        
        # Analyze missing
        self.analyze_missing_values(data)
        
        # Drop high missing features
        data = self.drop_high_missing_features(data, threshold=0.50)
        
        # Handle missing values
        data = self.handle_missing_values(data, method='median')
        
        # Feature engineering
        data = self.feature_engineering(data)
        
        # Remove outliers
        data = self.remove_outliers(data, target=target)
        
        # Encode categorical
        data = self.encode_categorical(data, fit=True)
        
        # Scale features
        data = self.scale_features(data, target=target, fit=True)
        
        logger.info(f"Final prepared data shape: {data.shape}")
        logger.info("=" * 80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 80)
        
        return data, properties, train