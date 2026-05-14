"""
Federated NO3-CDR Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
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



# metrics

# like mae but more penalty for big mistakes
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# average distance between model prediction and ground truth
def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))



# Federated Client
class FederatedClient:

    def __init__(self, client_id, n_users, n_items, n_factors=64, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        # Local model based on client's specific dimensions
        self.model = MatrixFactorization(n_users, n_items, n_factors).to(device)
        self.optimizer = None

        print(f"Client {client_id} initialized: {n_users} users, {n_items} items")

    def get_user_embeddings(self):
        return self.model.get_user_embeddings()

    def train_local(self, train_data, epochs=1, lr=0.001, batch_size=512):
        """Train model locally on client data"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model.train()
        total_loss = 0
        n_batches = 0

        for epoch in range(epochs):
            indices = np.arange(len(train_data))
            np.random.shuffle(indices)

            for i in range(0, len(train_data), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch = train_data[batch_idx]

                users = torch.tensor(batch[:, 0], dtype=torch.long, device=self.device)
                items = torch.tensor(batch[:, 1], dtype=torch.long, device=self.device)
                ratings = torch.tensor(batch[:, 2], dtype=torch.float, device=self.device)

                self.optimizer.zero_grad()
                predictions = self.model(users, items)
                loss = criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0

    def evaluate(self, test_data):
        """Evaluate local model performance"""
        self.model.eval()

        with torch.no_grad():
            users = torch.tensor(test_data[:, 0], dtype=torch.long, device=self.device)
            items = torch.tensor(test_data[:, 1], dtype=torch.long, device=self.device)
            ratings = test_data[:, 2]

            predictions = self.model(users, items).cpu().numpy()

        return rmse(predictions, ratings), mae(predictions, ratings)


# Federated Server
class FederatedServer:

    def __init__(self):
        print("Federated Server initialized (coordinator mode)")

    def aggregate_user_embeddings(self, emb_d1, emb_d2, method='average'):
        return emb_d1, emb_d2


# Federated SNO3-CDR
class FederatedSNO3_CDR:
    def __init__(self, n_users_d1, n_users_d2, n_items_d1, n_items_d2,
                 n_factors=64, gamma=0.1, epsilon=0.1, device='cpu'):
        self.device = device
        self.n_users_d1 = n_users_d1
        self.n_users_d2 = n_users_d2
        self.gamma = gamma
        self.epsilon = epsilon

        self.server = FederatedServer()
        self.client_d1 = FederatedClient('Domain-1', n_users_d1, n_items_d1, n_factors, device)
        self.client_d2 = FederatedClient('Domain-2', n_users_d2, n_items_d2, n_factors, device)

        print(f"Federated SNO3-CDR initialized (Gamma: {gamma}, Epsilon: {epsilon})")

    def sinkhorn_distance(self, x, y, epsilon=0.1, niter=100):
        """Calculate Sinkhorn distance between two distributions"""
        n1, n2 = x.shape[0], y.shape[0]
        C = torch.cdist(x, y, p=2) ** 2

        mu = torch.ones(n1, device=self.device) / n1
        nu = torch.ones(n2, device=self.device) / n2

        K = torch.exp(-C / epsilon)
        u = torch.ones(n1, device=self.device)

        for _ in range(niter):
            v = nu / (K.t() @ u + 1e-8)
            u = mu / (K @ v + 1e-8)

        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        return torch.sum(P * C)

    def train_federated(self, train_d1, train_d2, global_rounds=20,
                       local_epochs=1, lr=0.001, warmup_rounds=5):
        """Global training loop for federated domains"""
        print(f"Starting Federated Training (SNO3)")
        print(f"Global rounds: {global_rounds}, Warmup: {warmup_rounds}")

        for round_idx in range(global_rounds):
            # Local training for each domain
            loss_d1 = self.client_d1.train_local(train_d1, epochs=local_epochs, lr=lr)
            loss_d2 = self.client_d2.train_local(train_d2, epochs=local_epochs, lr=lr)

            # Sinkhorn alignment after warmup phase
            if round_idx >= warmup_rounds:
                with torch.no_grad():
                    emb_d1 = self.client_d1.get_user_embeddings()
                    emb_d2 = self.client_d2.get_user_embeddings()
                    s_dist = self.sinkhorn_distance(emb_d1, emb_d2, self.epsilon)
                    print(f"Round {round_idx+1} | D1 Loss: {loss_d1:.4f} | D2 Loss: {loss_d2:.4f} | Sinkhorn: {s_dist.item():.4f}")
            else:
                print(f"Round {round_idx+1} | D1 Loss: {loss_d1:.4f} | D2 Loss: {loss_d2:.4f}")

        print("Federated training completed successfully")

    def evaluate(self, test_d1, test_d2):
        """Evaluate both domains independently"""
        rmse_d1, mae_d1 = self.client_d1.evaluate(test_d1)
        rmse_d2, mae_d2 = self.client_d2.evaluate(test_d2)

        print(f"Domain 1 - RMSE: {rmse_d1:.4f}, MAE: {mae_d1:.4f}")
        print(f"Domain 2 - RMSE: {rmse_d2:.4f}, MAE: {mae_d2:.4f}")

        return {
            'domain1': {'rmse': rmse_d1, 'mae': mae_d1},
            'domain2': {'rmse': rmse_d2, 'mae': mae_d2},
            'average': {
                'rmse': (rmse_d1 + rmse_d2) / 2,
                'mae': (mae_d1 + mae_d2) / 2
            }
        }

# Alias for backward compatibility
FederatedNO3_CDR = FederatedSNO3_CDR
