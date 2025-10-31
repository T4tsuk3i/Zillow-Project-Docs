# MODEL TRAINING MODULES - PART 1

## Table of Contents
1. [ridge_lasso_elastic.py](#ridge_lasso_elasticpy)
2. [rf_xgb_lgbm.py](#rf_xgb_lgbmpy)

---

## ridge_lasso_elastic.py

### Module Purpose
GPU-accelerated PyTorch implementations of Ridge Regression, Lasso, and ElasticNet. These are linear regression models with L1/L2 regularization trained on GPU for speed.

### Complete Code Documentation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```
**Lines 1-8**: Imports
- **torch**: Deep learning framework, GPU support
- **torch.nn**: Neural network layers/models
- **torch.optim**: Optimizers (SGD, Adam, etc.)
- **numpy, pandas**: Data manipulation
- **logging**: Progress tracking
- **sklearn.metrics**: Evaluation metrics

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
**Line 10**: Detect GPU availability
- If GPU available: use cuda device
- Otherwise: fall back to CPU
- All tensors will be moved to this device

```python
class PyTorchLinearModels:
    """GPU-accelerated linear regression models"""
```
**Lines 12-13**: Class for linear models

```python
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.device = device
```
**Lines 14-17**: Initialize
- Setup logger
- Store device (cuda or cpu)
- All models will train on self.device

```python
    def create_ridge_model(self, input_size, alpha=1.0):
```
**Line 19**: Create Ridge regression model
- **input_size**: Number of features
- **alpha**: Regularization strength (L2 penalty)

```python
        """Create a simple linear model for Ridge regression"""
        class RidgeModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.linear = nn.Linear(input_size, 1)  # Single output (continuous value)
            
            def forward(self, x):
                return self.linear(x)
```
**Lines 20-27**: Define Ridge model architecture
- **nn.Module**: Base class for all neural network models
- **nn.Linear(input_size, 1)**: Linear layer (W×x + b)
- **forward()**: Forward pass (how data flows through model)
- Single output for regression

```python
        model = RidgeModel(input_size).to(self.device)
        self.alpha = alpha
        return model
```
**Lines 29-30**: Create model and move to GPU
- **.to(self.device)**: Move model weights to GPU memory
- **self.alpha**: Store regularization strength

```python
    def train_ridge(self, X_train, y_train, alpha=1.0, epochs=300, lr=0.01):
```
**Line 32**: Main method to train Ridge model
- **X_train, y_train**: Training features and targets
- **alpha**: L2 regularization strength
- **epochs**: Number of training iterations
- **lr**: Learning rate (step size for optimization)

```python
        """Train Ridge regression model on GPU"""
        self.logger.info(f"Training Ridge (α={alpha}) on {self.device}...")
        
        # Convert to PyTorch tensors and move to GPU
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(self.device).unsqueeze(1)
```
**Lines 33-37**: Convert data to GPU tensors
- **torch.tensor()**: Convert numpy/pandas to PyTorch tensor
- **dtype=torch.float32**: 32-bit floating point (GPU-optimized)
- **.to(self.device)**: Transfer to GPU
- **.unsqueeze(1)**: Add dimension: (N,) → (N,1)

```python
        model = self.create_ridge_model(X_train.shape[1], alpha)
        
        # L2 loss function (MSE)
        criterion = nn.MSELoss()
```
**Lines 39-41**: Create model and loss
- **MSELoss**: Mean Squared Error = sum((y_true - y_pred)²) / N
- This is minimized during training

```python
        # Optimizer (gradient descent)
        optimizer = optim.SGD(model.parameters(), lr=lr)
```
**Line 43**: Create optimizer
- **SGD**: Stochastic Gradient Descent
- **model.parameters()**: Weights and biases to optimize
- **lr=0.01**: Learning rate (how big steps to take)

```python
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = model(X_tensor)
```
**Lines 45-48**: Training loop
- **for epoch in range(epochs)**: Iterate 300 times
- **model(X_tensor)**: Forward pass (compute predictions)

```python
            # MSE loss
            mse_loss = criterion(y_pred, y_tensor)
            
            # L2 regularization (Ridge)
            l2_reg = torch.tensor(0.0).to(self.device)
            for param in model.parameters():
                l2_reg += torch.sum(param ** 2)
```
**Lines 50-55**: Calculate loss with regularization
- **mse_loss**: MSE of predictions
- **l2_reg**: Sum of squared weights (Ridge penalty)
- Larger weights = larger penalty

```python
            # Total loss = MSE + alpha * L2
            total_loss = mse_loss + alpha * l2_reg
```
**Line 56**: Combine MSE loss and regularization
- **alpha**: Controls regularization strength
- Higher alpha = stronger penalty on large weights

```python
            # Backward pass (compute gradients)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```
**Lines 58-61**: Optimization step
- **optimizer.zero_grad()**: Reset gradients (else they accumulate)
- **backward()**: Compute gradients using backpropagation
- **optimizer.step()**: Update weights using gradients

```python
            if epoch % 50 == 0:
                self.logger.info(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")
```
**Lines 63-64**: Log progress every 50 epochs
- **.item()**: Extract scalar value from tensor
- Show loss decreasing over time

```python
        self.logger.info(f"  ✓ Ridge training complete")
        return model
```
**Lines 65-66**: Return trained model

```python
    def predict(self, model, X):
```
**Line 68**: Make predictions with trained model
- **model**: Trained PyTorch model
- **X**: Features to predict (numpy array or DataFrame)

```python
        """Make predictions with trained model"""
        X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, 
                               dtype=torch.float32).to(self.device)
        
        model.eval()  # Set to evaluation mode (disables dropout, etc.)
        with torch.no_grad():  # Don't compute gradients (faster)
            predictions = model(X_tensor).cpu().numpy().flatten()
```
**Lines 69-74**: Generate predictions
- Convert X to tensor on GPU
- **model.eval()**: Set model to evaluation mode (no training-specific behavior)
- **torch.no_grad()**: Speed up by not tracking gradients
- **.cpu().numpy()**: Transfer results back to CPU and convert to numpy
- **.flatten()**: Convert (N,1) to (N,) shape

```python
        return predictions
```
**Line 75**: Return predictions as numpy array

```python
    def train_lasso(self, X_train, y_train, alpha=0.01, epochs=300, lr=0.01):
```
**Line 77**: Train Lasso (L1 regularization)
- **alpha**: L1 regularization strength
- Lower alpha default (0.01) because L1 is stronger than L2

```python
        """Train Lasso regression on GPU"""
        self.logger.info(f"Training Lasso (α={alpha}) on {self.device}...")
        
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        model = self.create_ridge_model(X_train.shape[1], alpha)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            y_pred = model(X_tensor)
            mse_loss = criterion(y_pred, y_tensor)
            
            # L1 regularization (Lasso) - sum of absolute values
            l1_reg = torch.tensor(0.0).to(self.device)
            for param in model.parameters():
                l1_reg += torch.sum(torch.abs(param))
```
**Lines 78-96**: Lasso training
- Similar to Ridge but uses **torch.abs(param)** instead of **param**²
- **L1**: Sum of absolute values (causes feature selection)
- Some weights become exactly 0 (feature elimination)

```python
            total_loss = mse_loss + alpha * l1_reg
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                self.logger.info(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        self.logger.info(f"  ✓ Lasso training complete")
        return model
```
**Lines 97-106**: Rest of training loop (same as Ridge)

```python
    def train_elasticnet(self, X_train, y_train, alpha=0.01, l1_ratio=0.5, epochs=300, lr=0.01):
```
**Line 108**: Train ElasticNet (combination of L1 and L2)
- **alpha**: Total regularization strength
- **l1_ratio**: 0.5 = 50% L1, 50% L2

```python
        """Train ElasticNet regression on GPU"""
        self.logger.info(f"Training ElasticNet (α={alpha}, l1_ratio={l1_ratio}) on {self.device}...")
        
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        model = self.create_ridge_model(X_train.shape[1], alpha)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            y_pred = model(X_tensor)
            mse_loss = criterion(y_pred, y_tensor)
            
            # Elastic Net = L1 + L2 combination
            l1_reg = torch.tensor(0.0).to(self.device)
            l2_reg = torch.tensor(0.0).to(self.device)
            for param in model.parameters():
                l1_reg += torch.sum(torch.abs(param))
                l2_reg += torch.sum(param ** 2)
```
**Lines 109-130**: ElasticNet combines both penalties
- **L1**: Promotes sparsity (zeros)
- **L2**: Prevents large weights
- Together = best of both worlds

```python
            total_loss = mse_loss + alpha * (l1_ratio * l1_reg + (1 - l1_ratio) * l2_reg)
```
**Line 131**: Weighted combination of penalties
- **l1_ratio * l1_reg**: L1 portion
- **(1-l1_ratio) * l2_reg**: L2 portion

```python
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                self.logger.info(f"  Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        self.logger.info(f"  ✓ ElasticNet training complete")
        return model
```
**Lines 133-140**: Rest of training loop

---

## rf_xgb_lgbm.py

### Module Purpose
Train tree-based ensemble models: Random Forest, XGBoost, and LightGBM with GPU acceleration where possible.

### Complete Code Documentation

```python
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import logging
import numpy as np
import pandas as pd
```
**Lines 1-6**: Imports
- **xgboost, lightgbm**: GPU-accelerated gradient boosting
- **RandomForestRegressor**: Sklearn's random forest (CPU only)
- **logging**: Progress tracking

```python
class TreeBasedModels:
    """Train tree-based ensemble models"""
```
**Lines 8-9**: Class for tree models

```python
    def __init__(self, gpu_available=True):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.gpu_available = gpu_available
```
**Lines 10-14**: Initialize
- Check if GPU available
- Setup logger
- All methods check gpu_available flag

```python
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=15):
```
**Line 16**: Train Random Forest
- **n_estimators**: Number of trees (more = better but slower)
- **max_depth**: Maximum tree depth (prevents overfitting)

```python
        """Train Random Forest on CPU"""
        self.logger.info(f"Training Random Forest ({n_estimators} trees)...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        model.fit(X_train, y_train)
```
**Lines 17-26**: Create and train model
- **n_jobs=-1**: Parallelize across all CPU cores
- **random_state=42**: Reproducibility

```python
        self.logger.info(f"  ✓ Random Forest trained")
        return model
```
**Lines 28-29**: Return trained model

```python
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
```
**Line 31**: Train XGBoost with GPU
- **X_val, y_val**: Optional validation set for early stopping

```python
        """Train XGBoost on GPU"""
        self.logger.info("Training XGBoost on GPU...")
        
        params = {
            'objective': 'reg:squarederror',      # Regression task
            'metric': 'mae',                      # Evaluate with MAE
            'tree_method': 'gpu_hist',            # GPU histogram-based tree building
            'gpu_id': 0,                          # Use first GPU
            'max_depth': 6,                       # Max tree depth
            'learning_rate': 0.05,                # Boosting step size
            'subsample': 0.8,                     # Use 80% of rows per tree
            'colsample_bytree': 0.8,              # Use 80% of columns per tree
            'seed': 42,                           # Reproducibility
        }
```
**Lines 32-45**: XGBoost hyperparameters
- **tree_method='gpu_hist'**: GPU-accelerated histogram building
- Hyperparameters tuned for balanced accuracy/speed

```python
        dtrain = xgb.DMatrix(X_train, label=y_train)
```
**Line 47**: Create XGBoost DMatrix
- **DMatrix**: XGBoost's optimized data format
- Compresses data for faster training on GPU

```python
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
```
**Lines 49-52**: Setup validation set
- Monitor performance on validation set during training
- Can enable early stopping

```python
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=evals if evals else None,
            verbose_eval=50 if evals else False,
        )
```
**Lines 54-60**: Train model
- **num_boost_round**: Number of boosting iterations (trees)
- **verbose_eval=50**: Print results every 50 rounds

```python
        self.logger.info(f"  ✓ XGBoost trained")
        return model
```
**Lines 62-63**: Return trained model

```python
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
```
**Line 65**: Train LightGBM with GPU
- Similar to XGBoost but often faster

```python
        """Train LightGBM on GPU"""
        self.logger.info("Training LightGBM on GPU...")
        
        params = {
            'objective': 'regression',            # Regression task
            'metric': 'mae',                      # Evaluate with MAE
            'device': 'gpu',                      # Use GPU
            'gpu_device_id': 0,                   # First GPU
            'max_depth': 7,                       # Max tree depth
            'num_leaves': 31,                     # Max leaves per tree
            'learning_rate': 0.05,                # Boosting step size
            'subsample': 0.8,                     # Row sampling
            'colsample_bytree': 0.8,              # Column sampling
            'verbose': -1,                        # No progress output
            'seed': 42,                           # Reproducibility
        }
```
**Lines 66-80**: LightGBM hyperparameters
- Similar to XGBoost but with different naming
- **num_leaves**: LightGBM-specific parameter (max leaves)

```python
        train_data = lgb.Dataset(X_train, label=y_train)
```
**Line 82**: Create LightGBM Dataset

```python
        callbacks = []
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks = [
                lgb.early_stopping(50),  # Stop if no improvement for 50 rounds
                lgb.log_evaluation(50),  # Print eval results every 50 rounds
            ]
```
**Lines 84-89**: Setup callbacks
- **early_stopping**: Stop training if validation doesn't improve
- **log_evaluation**: Print progress

```python
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data] if X_val is not None else None,
            callbacks=callbacks,
        )
```
**Lines 91-96**: Train model

```python
        self.logger.info(f"  ✓ LightGBM trained")
        return model
```
**Lines 98-99**: Return trained model

```python
    def evaluate_model(self, model, X_test, y_test, model_type='xgboost'):
```
**Line 101**: Evaluate trained model
- **model_type**: Type of model (xgboost, lightgbm, or sklearn)

```python
        """Evaluate model performance"""
        self.logger.info(f"Evaluating model...")
        
        if model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            predictions = model.predict(dtest)
        elif model_type == 'lightgbm':
            predictions = model.predict(X_test)
        else:  # sklearn models
            predictions = model.predict(X_test)
```
**Lines 102-111**: Get predictions based on model type

```python
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        self.logger.info(f"  MAE: {mae:.6f}")
        self.logger.info(f"  RMSE: {rmse:.6f}")
        self.logger.info(f"  R²: {r2:.6f}")
```
**Lines 113-120**: Calculate and log metrics

```python
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
```
**Line 121**: Return metrics dictionary

---

End of Model Training Modules Part 1
