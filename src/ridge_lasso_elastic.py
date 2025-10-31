"""
GPU-Accelerated Linear Models Module (PyTorch)
Ridge, Lasso, ElasticNet with CUDA Support
Optimized for RTX 3050 with CUDA 13
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Detect GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class LinearRegressionTorch(nn.Module):
    """Simple linear regression model in PyTorch"""
    def __init__(self, n_features):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class PyTorchLinearModels:
    """GPU-accelerated linear models using PyTorch"""
    
    def __init__(self):
        self.models = {}
        self.device = device
        self.training_history = {}
    
    def prepare_data(self, X, y, batch_size=256):
        """Convert numpy arrays to PyTorch tensors and create data loader"""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        return loader, X_tensor, y_tensor
    
    def train_ridge(self, X_train, y_train, alpha=1.0, epochs=500, lr=0.01, batch_size=256):
        """Train Ridge Regression on GPU (L2 regularization)"""
        logger.info(f"Training Ridge Regression on {self.device} (alpha={alpha}, epochs={epochs})")
        
        loader, _, _ = self.prepare_data(X_train, y_train, batch_size)
        
        model = LinearRegressionTorch(X_train.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                
                # MSE loss
                mse_loss = criterion(outputs, batch_y)
                
                # L2 regularization (Ridge)
                l2_reg = torch.tensor(0.0, device=self.device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                
                # Total loss
                loss = mse_loss + alpha * l2_reg
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        self.models['Ridge'] = model
        self.training_history['Ridge'] = losses
        logger.info(f"✓ Ridge training complete on {self.device}")
        
        return model
    
    def train_lasso(self, X_train, y_train, alpha=0.01, epochs=500, lr=0.01, batch_size=256):
        """Train Lasso Regression on GPU (L1 regularization)"""
        logger.info(f"Training Lasso Regression on {self.device} (alpha={alpha}, epochs={epochs})")
        
        loader, _, _ = self.prepare_data(X_train, y_train, batch_size)
        
        model = LinearRegressionTorch(X_train.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                
                # MSE loss
                mse_loss = criterion(outputs, batch_y)
                
                # L1 regularization (Lasso)
                l1_reg = torch.tensor(0.0, device=self.device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, p=1)
                
                # Total loss
                loss = mse_loss + alpha * l1_reg
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        self.models['Lasso'] = model
        self.training_history['Lasso'] = losses
        logger.info(f"✓ Lasso training complete on {self.device}")
        
        return model
    
    def train_elasticnet(self, X_train, y_train, alpha=0.01, l1_ratio=0.5, epochs=500, lr=0.01, batch_size=256):
        """Train ElasticNet Regression on GPU (L1 + L2 regularization)"""
        logger.info(f"Training ElasticNet on {self.device} (alpha={alpha}, l1_ratio={l1_ratio}, epochs={epochs})")
        
        loader, _, _ = self.prepare_data(X_train, y_train, batch_size)
        
        model = LinearRegressionTorch(X_train.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x)
                
                # MSE loss
                mse_loss = criterion(outputs, batch_y)
                
                # ElasticNet regularization (L1 + L2)
                l1_reg = torch.tensor(0.0, device=self.device)
                l2_reg = torch.tensor(0.0, device=self.device)
                
                for param in model.parameters():
                    l1_reg += torch.norm(param, p=1)
                    l2_reg += torch.norm(param, p=2)
                
                # Total loss with combined regularization
                reg = alpha * (l1_ratio * l1_reg + (1 - l1_ratio) * l2_reg)
                loss = mse_loss + reg
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        self.models['ElasticNet'] = model
        self.training_history['ElasticNet'] = losses
        logger.info(f"✓ ElasticNet training complete on {self.device}")
        
        return model
    
    def predict(self, model, X_test):
        """Make predictions on GPU"""
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy().flatten()
        
        return predictions
    
    def get_weights(self, model):
        """Extract weights from model"""
        with torch.no_grad():
            weights = model.linear.weight.cpu().numpy().flatten()
            bias = model.linear.bias.cpu().numpy()[0]
        
        return weights, bias