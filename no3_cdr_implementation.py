"""
Dual-Target Disjointed Cross-Domain Recommendation
Implementation of SNO3-CDR algorithm
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
import warnings

# base model for recomendation system
class MatrixFactorization(nn.Module):

    def __init__(self, n_users, n_items, n_factors=64):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
    
    def forward(self, user_ids, item_ids):
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        
        dot_product = (user_embedding * item_embedding).sum(dim=1)
        prediction = dot_product + user_bias + item_bias + self.global_bias
        
        return prediction
    
    def get_user_embeddings(self):
        return self.user_factors.weight.detach()


# Sinkhorn Distance
def sinkhorn_distance(x, y, epsilon=0.1, niter=100, p=2):
    """
    Compute Sinkhorn distance between two point clouds
    
    Args:
        x: First point cloud (n1 x d)
        y: Second point cloud (n2 x d)
        epsilon: Regularization parameter
        niter: Number of iterations
        p: Power for distance (2 for Euclidean)
    
    Returns:
        Sinkhorn distance
    """
    n1, n2 = x.shape[0], y.shape[0] # number of users in each domain
    
    # Compute cost matrix (Euclidean distance)
    C = torch.cdist(x, y, p=p) ** p
    
    # Initialize dual variables
    mu = torch.ones(n1, device=x.device) / n1  # for n=3 -> mu=[1/3, 1/3, 1/3]
    nu = torch.ones(n2, device=y.device) / n2
    
    # Sinkhorn iterations
    K = torch.exp(-C / epsilon)
    u = torch.ones(n1, device=x.device)
    
    for _ in range(niter):
        v = nu / (K.t() @ u + 1e-8) # for columns
        u = mu / (K @ v + 1e-8) # for rows
    
    # Transport plan
    P = u.unsqueeze(1) * K * v.unsqueeze(0) # for amend matrices shape that can be *
    
    # Sinkhorn distance
    distance = torch.sum(P * C)
    
    return distance


def bidirectional_sinkhorn_distance(x, y, epsilon=0.1, niter=100):
    """Bi-directional Sinkhorn distance"""
    d1 = sinkhorn_distance(x, y, epsilon, niter)
    d2 = sinkhorn_distance(y, x, epsilon, niter)
    return d1 + d2


# HNO3-CDR: Hard Matching
class HNO3_CDR:
    """Hard matching cross-domain recommendation using Hungarian algorithm"""
    
    def __init__(self, model_class, n_users_d1, n_users_d2, n_items_d1, 
                 n_items_d2, n_factors=64, device='cuda'):
        self.device = device
        self.n_users_d1 = n_users_d1
        self.n_users_d2 = n_users_d2
        self.n_items_d1 = n_items_d1
        self.n_items_d2 = n_items_d2
        
        # Total users and items
        self.n_users = n_users_d1 + n_users_d2
        self.n_items = n_items_d1 + n_items_d2
        
        # Initialize models
        self.model_class = model_class
        self.model_stage1 = model_class(self.n_users, self.n_items, n_factors).to(device)
        self.model_stage2 = None
        
    def train_stage1(self, train_data_d1, train_data_d2, epochs=20, lr=0.001, task='rating'):
        """Stage 1: Train on combined data to get user representations"""
        optimizer = optim.Adam(self.model_stage1.parameters(), lr=lr)
        
        if task == 'rating':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
        
        self.model_stage1.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Train on domain 1
            for batch in self._create_batches(train_data_d1):
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device).float()
                
                optimizer.zero_grad()
                predictions = self.model_stage1(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Train on domain 2 (adjust indices)
            for batch in self._create_batches(train_data_d2, offset_user=self.n_users_d1, 
                                             offset_item=self.n_items_d1):
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device).float()
                
                optimizer.zero_grad()
                predictions = self.model_stage1(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Stage 1 - Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    def hungarian_matching(self):
        """Stage 2: Apply Hungarian algorithm for user matching"""
        # Get user embeddings
        user_embeddings = self.model_stage1.get_user_embeddings().cpu().numpy()
        
        # Split embeddings by domain
        emb_d1 = user_embeddings[:self.n_users_d1]
        emb_d2 = user_embeddings[self.n_users_d1:self.n_users_d1 + self.n_users_d2]
        
        # Compute cost matrix (negative cosine similarity)
        cost_matrix = -np.dot(emb_d1, emb_d2.T) / (
            np.linalg.norm(emb_d1, axis=1, keepdims=True) * 
            np.linalg.norm(emb_d2, axis=1, keepdims=True).T + 1e-8
        )
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return row_ind, col_ind
    
    def train_stage2(self, train_data_d1, train_data_d2, row_ind, col_ind, 
                     epochs=20, lr=0.001, task='rating'):
        """Stage 3: Train with matched users"""
        # Create mapping
        user_mapping_d2_to_d1 = {}
        for i, j in zip(row_ind, col_ind):
            user_mapping_d2_to_d1[j] = i
        
        # Initialize new model with matched users
        n_matched_users = max(self.n_users_d1, self.n_users_d2)
        self.model_stage2 = self.model_class(n_matched_users, self.n_items, 64).to(self.device)
        
        optimizer = optim.Adam(self.model_stage2.parameters(), lr=lr)
        
        if task == 'rating':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
        
        self.model_stage2.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Train on domain 1
            for batch in self._create_batches(train_data_d1):
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device).float()
                
                optimizer.zero_grad()
                predictions = self.model_stage2(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Train on domain 2 with mapped users
            for batch in self._create_batches(train_data_d2, offset_item=self.n_items_d1):
                user_ids, item_ids, ratings = batch
                
                # Map users
                mapped_users = torch.tensor([user_mapping_d2_to_d1.get(u.item(), u.item()) 
                                            for u in user_ids], device=self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device).float()
                
                optimizer.zero_grad()
                predictions = self.model_stage2(mapped_users, item_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Stage 2 - Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    def _create_batches(self, data, batch_size=1024, offset_user=0, offset_item=0):
        """Create batches from data"""
        n_samples = len(data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_data = data[batch_indices]
            user_ids = torch.tensor(batch_data[:, 0] + offset_user, dtype=torch.long)
            item_ids = torch.tensor(batch_data[:, 1] + offset_item, dtype=torch.long)
            ratings = torch.tensor(batch_data[:, 2], dtype=torch.float)
            
            yield user_ids, item_ids, ratings
    
    def predict(self, user_ids, item_ids, domain=1):
        """Make predictions"""
        model = self.model_stage2 if self.model_stage2 is not None else self.model_stage1
        model.eval()
        
        with torch.no_grad():
            if domain == 2 and self.model_stage2 is None:
                user_ids = user_ids + self.n_users_d1
                item_ids = item_ids + self.n_items_d1
            elif domain == 2:
                item_ids = item_ids + self.n_items_d1
            
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
            item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            
            predictions = model(user_ids, item_ids)
        
        return predictions.cpu().numpy()


# SNO3-CDR: Soft Matching
class SNO3_CDR:
    """Soft matching cross-domain recommendation using Sinkhorn distance"""
    
    def __init__(self, model_class, n_users_d1, n_users_d2, n_items_d1, 
                 n_items_d2, n_factors=64, gamma=0.1, epsilon=0.1, device='cuda'):
        self.device = device
        self.n_users_d1 = n_users_d1
        self.n_users_d2 = n_users_d2
        self.n_items_d1 = n_items_d1
        self.n_items_d2 = n_items_d2
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Total users and items
        self.n_users = n_users_d1 + n_users_d2
        self.n_items = n_items_d1 + n_items_d2
        
        # Initialize model
        self.model = model_class(self.n_users, self.n_items, n_factors).to(device)
    
    def train(self, train_data_d1, train_data_d2, epochs=20, lr=0.001, 
              warmup_epochs=5, task='rating'):
        """End-to-end training with Sinkhorn regularization"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        if task == 'rating':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_rec_loss = 0
            total_sink_loss = 0
            
            # Train on domain 1
            for batch in self._create_batches(train_data_d1):
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device).float()
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                rec_loss = criterion(predictions, ratings)
                
                # Add Sinkhorn loss after warmup
                if epoch >= warmup_epochs:
                    user_emb_d1 = self.model.get_user_embeddings()[:self.n_users_d1]
                    user_emb_d2 = self.model.get_user_embeddings()[self.n_users_d1:self.n_users_d1+self.n_users_d2]
                    sink_loss = bidirectional_sinkhorn_distance(user_emb_d1, user_emb_d2, 
                                                               self.epsilon)
                    loss = rec_loss + self.gamma * sink_loss
                    total_sink_loss += sink_loss.item()
                else:
                    loss = rec_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
            
            # Train on domain 2
            for batch in self._create_batches(train_data_d2, offset_user=self.n_users_d1, 
                                             offset_item=self.n_items_d1):
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device).float()
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids)
                rec_loss = criterion(predictions, ratings)
                
                # Add Sinkhorn loss after warmup
                if epoch >= warmup_epochs:
                    user_emb_d1 = self.model.get_user_embeddings()[:self.n_users_d1]
                    user_emb_d2 = self.model.get_user_embeddings()[self.n_users_d1:self.n_users_d1+self.n_users_d2]
                    sink_loss = bidirectional_sinkhorn_distance(user_emb_d1, user_emb_d2, 
                                                               self.epsilon)
                    loss = rec_loss + self.gamma * sink_loss
                    total_sink_loss += sink_loss.item()
                else:
                    loss = rec_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
            
            if (epoch + 1) % 5 == 0:
                if epoch >= warmup_epochs:
                    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}, "
                          f"Rec Loss: {total_rec_loss:.4f}, Sink Loss: {total_sink_loss:.4f}")
                else:
                    print(f"Warmup Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    def _create_batches(self, data, batch_size=1024, offset_user=0, offset_item=0):
        """Create batches from data"""
        n_samples = len(data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_data = data[batch_indices]
            user_ids = torch.tensor(batch_data[:, 0] + offset_user, dtype=torch.long)
            item_ids = torch.tensor(batch_data[:, 1] + offset_item, dtype=torch.long)
            ratings = torch.tensor(batch_data[:, 2], dtype=torch.float)
            
            yield user_ids, item_ids, ratings
    
    def predict(self, user_ids, item_ids, domain=1):
        """Make predictions"""
        self.model.eval()
        
        with torch.no_grad():
            if domain == 2:
                user_ids = user_ids + self.n_users_d1
                item_ids = item_ids + self.n_items_d1
            
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
            item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            
            predictions = self.model(user_ids, item_ids)
        
        return predictions.cpu().numpy()


#  Evaluation Metrics
def rmse(predictions, targets):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def mae(predictions, targets):
    """Mean Absolute Error"""
    return np.mean(np.abs(predictions - targets))